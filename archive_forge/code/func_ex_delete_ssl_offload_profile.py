from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_delete_ssl_offload_profile(self, profile_id):
    """
        Delete an offload profile

        :param profile_id: Id of profile to be deleted
        :type profile_id: ``str``
        :returns: ``bool``
        """
    del_profile_elem = ET.Element('deleteSslOffloadProfile', {'id': profile_id, 'xmlns': TYPES_URN})
    result = self.connection.request_with_orgId_api_2('networkDomainVip/deleteSslOffloadProfile', method='POST', data=ET.tostring(del_profile_elem)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']