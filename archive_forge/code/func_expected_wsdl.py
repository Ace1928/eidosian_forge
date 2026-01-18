import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def expected_wsdl(version):
    driver_abs_dir = os.path.abspath(os.path.dirname(pbm.__file__))
    path = os.path.join(driver_abs_dir, 'wsdl', version, 'pbmService.wsdl')
    return urlparse.urljoin('file:', urllib.pathname2url(path))