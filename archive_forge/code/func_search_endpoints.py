from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def search_endpoints(self, id=None, filters=None):
    """List Keystone endpoints.

        :param id: ID of endpoint(s).
        :param filters: dictionary of meta data to use for further filtering.
            Elements of this dictionary may, themselves, be dictionaries.
            Example::

                {
                  'last_name': 'Smith',
                  'other': {
                      'gender': 'Female'
                  }
                }

            OR

            A string containing a jmespath expression for further filtering.
            Example::

                "[?last_name==`Smith`] | [?other.gender]==`Female`]"

        :returns: A list of identity ``Endpoint`` objects
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    endpoints = self.list_endpoints()
    return _utils._filter_list(endpoints, id, filters)