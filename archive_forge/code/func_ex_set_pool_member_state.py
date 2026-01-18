from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_set_pool_member_state(self, member, enabled=True):
    request = ET.Element('editPoolMember', {'xmlns': TYPES_URN, 'id': member.id})
    state = 'ENABLED' if enabled is True else 'DISABLED'
    ET.SubElement(request, 'status').text = state
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/editPoolMember', method='POST', data=ET.tostring(request)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']