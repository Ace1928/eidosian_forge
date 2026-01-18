import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def requires_floating_ip(self):
    """Return whether or not this cloud requires floating ips.


        :returns: True of False if know, None if discovery is needed.
                  If requires_floating_ip is not configured but the cloud is
                  known to not provide floating ips, will return False.
        """
    if self.config['floating_ip_source'] == 'None':
        return False
    return self.config.get('requires_floating_ip')