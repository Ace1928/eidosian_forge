import os
import re
import tempfile
from os_win import _utils
from os_win import constants
from os_win.tests.functional import test_base
from os_win import utilsfactory
def test_acls(self):
    tmp_suffix = 'oswin-func-test'
    tmp_dir = tempfile.mkdtemp(suffix=tmp_suffix)
    self.addCleanup(self._pathutils.rmtree, tmp_dir)
    tmp_file_paths = []
    for idx in range(2):
        tmp_file_path = os.path.join(tmp_dir, 'tmp_file_%s' % idx)
        with open(tmp_file_path, 'w') as f:
            f.write('test')
        tmp_file_paths.append(tmp_file_path)
    trustee = 'NULL SID'
    self._pathutils.add_acl_rule(path=tmp_dir, trustee_name=trustee, access_rights=constants.ACE_GENERIC_READ, access_mode=constants.ACE_GRANT_ACCESS, inheritance_flags=constants.ACE_OBJECT_INHERIT | constants.ACE_CONTAINER_INHERIT)
    self._pathutils.add_acl_rule(path=tmp_file_paths[0], trustee_name=trustee, access_rights=constants.ACE_GENERIC_WRITE, access_mode=constants.ACE_GRANT_ACCESS)
    self._pathutils.copy_acls(tmp_file_paths[0], tmp_file_paths[1])
    self._assert_contains_ace(tmp_dir, trustee, '(OI)(CI).*(GR)')
    for path in tmp_file_paths:
        self._assert_contains_ace(path, trustee, '(W,Rc)')