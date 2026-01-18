from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import os
import re
import subprocess
import textwrap
import six
from six.moves import zip
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite.messages import DecodeError
from boto import config
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import GetFailureCount
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import IamChOnResourceWithConditionsException
from gslib.help_provider import CreateHelpText
from gslib.metrics import LogCommandParams
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.storage_url import GetSchemeFromUrlString
from gslib.storage_url import IsKnownUrlScheme
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import IAM_POLICY_VERSION
from gslib.utils.constants import NO_MAX
from gslib.utils import iam_helper
from gslib.utils.iam_helper import BindingStringToTuple
from gslib.utils.iam_helper import BindingsTuple
from gslib.utils.iam_helper import DeserializeBindingsTuple
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.iam_helper import SerializeBindingsTuple
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.shim_util import GcloudStorageFlag
def run_gcloud_storage(self):
    if self.sub_command != 'ch':
        return super().run_gcloud_storage()
    self.ParseSubOpts()
    bindings_tuples, patterns = self._GetSettingsAndDiffs()
    resource_type = self._get_resource_type(patterns)
    list_settings = []
    if self.recursion_requested:
        list_settings.append('-r')
    return_code = 0
    for url_pattern in patterns:
        self._RaiseIfInvalidUrl(url_pattern)
        if resource_type == 'objects':
            ls_process = self._run_ch_subprocess(['storage', 'ls', '--json'] + list_settings + [str(url_pattern)])
            if ls_process.returncode:
                return_code = 1
                continue
            ls_output = json.loads(ls_process.stdout)
            urls = [resource['url'] for resource in ls_output if resource['type'] == 'cloud_object']
        else:
            urls = [str(url_pattern)]
        for url in urls:
            get_process = self._run_ch_subprocess(['storage', resource_type, 'get-iam-policy', url, '--format=json'])
            if get_process.returncode:
                return_code = 1
                continue
            policy = json.loads(get_process.stdout)
            bindings = iam_helper.BindingsDictToUpdateDict(policy['bindings'])
            for is_grant, diff in bindings_tuples:
                diff_dict = iam_helper.BindingsDictToUpdateDict(diff)
                bindings = PatchBindings(bindings, diff_dict, is_grant)
            policy['bindings'] = [{'role': r, 'members': sorted(list(m))} for r, m in sorted(six.iteritems(bindings))]
            set_process = self._run_ch_subprocess(['storage', resource_type, 'set-iam-policy', url, '-'], stdin=json.dumps(policy, sort_keys=True))
            if set_process.returncode:
                return_code = 1
    return return_code