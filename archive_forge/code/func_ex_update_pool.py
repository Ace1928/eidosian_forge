from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_update_pool(self, pool):
    """
        Update the properties of an existing pool
        only method, serviceDownAction and slowRampTime are updated

        :param pool: The instance of ``NttCisPool`` to update
        :type  pool: ``NttCisPool``

        :return: ``True`` for success, ``False`` for failure
        :rtype: ``bool``
        """
    create_node_elm = ET.Element('editPool', {'xmlns': TYPES_URN})
    create_node_elm.set('id', pool.id)
    ET.SubElement(create_node_elm, 'loadBalanceMethod').text = str(pool.load_balance_method)
    ET.SubElement(create_node_elm, 'healthMonitorId').text = pool.health_monitor_id
    ET.SubElement(create_node_elm, 'serviceDownAction').text = pool.service_down_action
    ET.SubElement(create_node_elm, 'slowRampTime').text = str(pool.slow_ramp_time)
    response = self.connection.request_with_orgId_api_2(action='networkDomainVip/editPool', method='POST', data=ET.tostring(create_node_elm)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']