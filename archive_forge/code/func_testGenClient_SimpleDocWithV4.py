import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
def testGenClient_SimpleDocWithV4(self):
    with test_utils.TempDir() as tmp_dir_path:
        gen_client.main([gen_client.__file__, '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--apitools_version', '0.4.12', '--root_package', 'google.apis', 'client'])
        self.assertEquals(set(['dns_v1_client.py', 'dns_v1_messages.py', '__init__.py']), set(os.listdir(tmp_dir_path)))