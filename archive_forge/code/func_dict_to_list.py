from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.ospfv2.ospfv2 import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv2 import (
def dict_to_list(self, ospf_data):
    """Converts areas, interfaces in each process to list
        :param ospf_data: ospf data
        :rtype: dictionary
        :returns: facts_output
        """
    facts_output = {'processes': []}
    for process in ospf_data.get('processes', []):
        if 'passive_interfaces' in process and process['passive_interfaces'].get('default'):
            if process.get('passive_interfaces', {}).get('interface'):
                process['passive_interfaces']['interface']['name'] = [each for each in process['passive_interfaces']['interface']['name'] if each]
        if 'areas' in process:
            process['areas'] = list(process['areas'].values())
        facts_output['processes'].append(process)
    return facts_output