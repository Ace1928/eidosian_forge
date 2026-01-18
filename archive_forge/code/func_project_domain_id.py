import abc
import urllib.parse
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
@project_domain_id.setter
def project_domain_id(self, value):
    self._project_domain_id = value