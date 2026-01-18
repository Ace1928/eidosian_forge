from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_destroy_pool(self, pool):
    """
        Destroy an existing pool

        :param pool: The instance of ``NttCisPool`` to destroy
        :type  pool: ``NttCisPool``

        :return: ``True`` for success, ``False`` for failure
        :rtype: ``bool``
        """
    destroy_request = ET.Element('deletePool', {'xmlns': TYPES_URN, 'id': pool.id})
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/deletePool', method='POST', data=ET.tostring(destroy_request)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']