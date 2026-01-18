import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_pbm_wsdl_location(self):
    wsdl = pbm.get_pbm_wsdl_location(None)
    self.assertIsNone(wsdl)

    def expected_wsdl(version):
        driver_abs_dir = os.path.abspath(os.path.dirname(pbm.__file__))
        path = os.path.join(driver_abs_dir, 'wsdl', version, 'pbmService.wsdl')
        return urlparse.urljoin('file:', urllib.pathname2url(path))
    with mock.patch('os.path.exists') as path_exists:
        path_exists.return_value = True
        wsdl = pbm.get_pbm_wsdl_location('5')
        self.assertEqual(expected_wsdl('5'), wsdl)
        wsdl = pbm.get_pbm_wsdl_location('5.5')
        self.assertEqual(expected_wsdl('5.5'), wsdl)
        wsdl = pbm.get_pbm_wsdl_location('5.5.1')
        self.assertEqual(expected_wsdl('5.5'), wsdl)
        path_exists.return_value = False
        wsdl = pbm.get_pbm_wsdl_location('5.5')
        self.assertIsNone(wsdl)