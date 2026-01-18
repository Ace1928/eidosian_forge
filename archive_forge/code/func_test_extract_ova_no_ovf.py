import os.path
import shutil
import tarfile
import tempfile
from unittest import mock
from defusedxml.ElementTree import ParseError
from glance.async_.flows import ovf_process
import glance.tests.utils as test_utils
from oslo_config import cfg
def test_extract_ova_no_ovf(self):
    ova_file_path = os.path.join(self.test_ova_dir, 'testserver-no-ovf.ova')
    iextractor = ovf_process.OVAImageExtractor()
    with open(ova_file_path, 'rb') as ova_file:
        self.assertRaises(RuntimeError, iextractor.extract, ova_file)