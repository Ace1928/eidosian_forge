from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.swarm import AnsibleDockerSwarmClient
from ansible_collections.community.docker.plugins.module_utils.common import RequestException
from ansible_collections.community.docker.plugins.module_utils.util import (
@staticmethod
def get_essential_facts_nodes(item):
    object_essentials = dict()
    object_essentials['ID'] = item.get('ID')
    object_essentials['Hostname'] = item['Description']['Hostname']
    object_essentials['Status'] = item['Status']['State']
    object_essentials['Availability'] = item['Spec']['Availability']
    if 'ManagerStatus' in item:
        object_essentials['ManagerStatus'] = item['ManagerStatus']['Reachability']
        if 'Leader' in item['ManagerStatus'] and item['ManagerStatus']['Leader'] is True:
            object_essentials['ManagerStatus'] = 'Leader'
    else:
        object_essentials['ManagerStatus'] = None
    object_essentials['EngineVersion'] = item['Description']['Engine']['EngineVersion']
    return object_essentials