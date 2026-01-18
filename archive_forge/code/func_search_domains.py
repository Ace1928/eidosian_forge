from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def search_domains(self, filters=None, name_or_id=None):
    """Search Keystone domains.

        :param name_or_id: Name or ID of the domain(s).
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

        :returns: a list of identity ``Domain`` objects
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    if filters is None:
        filters = {}
    if name_or_id is not None:
        domains = self.list_domains()
        return _utils._filter_list(domains, name_or_id, filters)
    else:
        return self.list_domains(**filters)