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
def patch_binding(policy, role, new_policy):
    """Returns a patched Python object representation of a Policy.

  Given replaces the original role:members binding in policy with new_policy.

  Args:
    policy: Python dict representation of a Policy instance.
    role: An IAM policy role (e.g. "roles/storage.objectViewer"). Fully
          specified in BindingsValueListEntry.
    new_policy: A Python dict representation of a Policy instance, with a
                single BindingsValueListEntry entry.

  Returns:
    A Python dict representation of the patched IAM Policy object.
  """
    bindings = [b for b in policy.get('bindings', []) if b.get('role', '') != role]
    bindings.extend(new_policy)
    policy = dict(policy)
    policy['bindings'] = bindings
    return policy