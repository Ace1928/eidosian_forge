import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
def testGenClient_SimpleDocEmptyInit(self):
    with test_utils.TempDir() as tmp_dir_path:
        gen_client.main([gen_client.__file__, '--init-file', 'empty', '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'google.apis', 'client'])
        expected_files = set(['dns_v1_client.py', 'dns_v1_messages.py', '__init__.py'])
        self.assertEquals(expected_files, set(os.listdir(tmp_dir_path)))
        init_file = _GetContent(os.path.join(tmp_dir_path, '__init__.py'))
        self.assertEqual('"""Package marker file."""\n\nfrom __future__ import absolute_import\n\nimport pkgutil\n\n__path__ = pkgutil.extend_path(__path__, __name__)\n', init_file)