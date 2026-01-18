from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import time
from gslib.command import CreateOrGetGsutilLogger
from gslib.tab_complete import CloudObjectCompleter
from gslib.tab_complete import TAB_COMPLETE_CACHE_TTL
from gslib.tab_complete import TabCompletionCache
import gslib.tests.testcase as testcase
from gslib.tests.util import ARGCOMPLETE_AVAILABLE
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.boto_util import GetTabCompletionCacheFilename
def test_acl_argument(self):
    """Tests tab completion for ACL arguments."""
    local_file = 'a_local_file'
    local_dir = self.CreateTempDir(test_files=[local_file])
    local_file_request = '%s%s' % (local_dir, os.sep)
    expected_local_file_result = '%s ' % os.path.join(local_dir, local_file)
    self.RunGsUtilTabCompletion(['acl', 'set', local_file_request], expected_results=[expected_local_file_result])
    self.RunGsUtilTabCompletion(['acl', 'set', 'priv'], expected_results=['private '])
    local_file = 'priv_file'
    local_dir = self.CreateTempDir(test_files=[local_file])
    with WorkingDirectory(local_dir):
        self.RunGsUtilTabCompletion(['acl', 'set', 'priv'], expected_results=[local_file, 'private'])