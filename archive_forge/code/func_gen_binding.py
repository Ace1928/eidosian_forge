from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
def gen_binding(role, members=None, condition=None):
    """Generate the "bindings" portion of an IAM Policy dictionary.

  Generates list of dicts which each represent a
  storage_v1_messages.Policy.BindingsValueListEntry object. The list will
  contain a single dict which has attributes corresponding to arguments passed
  to this method.

  Args:
    role: (str) An IAM policy role (e.g. "roles/storage.objectViewer"). Fully
        specified in BindingsValueListEntry.
    members: (List[str]) A list of members (e.g. ["user:foo@bar.com"]). If None,
        bind to ["allUsers"]. Fully specified in BindingsValueListEntry.
    condition: (Dict) A dictionary representing the JSON used to define a
        binding condition, containing the keys "description", "expression", and
        "title".

  Returns:
    (List[Dict[str, Any]]) A Python representation of the "bindings" portion of
    an IAM Policy.
  """
    binding = {'members': ['allUsers'] if members is None else members, 'role': role}
    if condition:
        binding['condition'] = condition
    return [binding]