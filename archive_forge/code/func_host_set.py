from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
def host_set(self, host=None, status=None, maintenance_mode=None, **params):
    """Modify host properties

        https://docs.openstack.org/api-ref/compute/#update-host-status
        Valid for Compute 2.0 - 2.42

        status
        maintenance_mode
        """
    url = '/os-hosts'
    params = {}
    if status:
        params['status'] = status
    if maintenance_mode:
        params['maintenance_mode'] = maintenance_mode
    if params == {}:
        return None
    else:
        return self._request('PUT', '/%s/%s' % (url, host), json=params).json()