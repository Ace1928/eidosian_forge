from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def convert_list_to_dict(self, *args, **kwargs):
    temp = {}
    if kwargs['val'].get('services_object'):
        for every in kwargs['val']['services_object']:
            temp_key = every['protocol']
            if 'source_port' in every:
                if 'range' in every['source_port']:
                    temp_key = 'range' + '_' + str(every['source_port']['range']['start']) + '_' + str(every['source_port']['range']['end'])
                else:
                    source_key = list(every['source_port'])[0]
                    temp_key = temp_key + '_' + source_key + '_' + every['source_port'][source_key]
            if 'destination_port' in every:
                if 'range' in every['destination_port']:
                    temp_key = 'range' + '_' + str(every['destination_port']['range']['start']) + '_' + str(every['destination_port']['range']['end'])
                else:
                    destination_key = list(every['destination_port'])[0]
                    temp_key = temp_key + '_' + destination_key + '_' + every['destination_port'][destination_key]
            temp.update({temp_key: every})
        return temp
    elif kwargs['val'].get('port_object'):
        for every in kwargs['val']['port_object']:
            if 'range' in every:
                temp_key = 'start' + '_' + every['range']['start'] + '_' + 'end' + '_' + every['range']['end']
            else:
                every_key = list(every)[0]
                temp_key = every_key + '_' + every[every_key]
            temp.update({temp_key: every})
        return temp