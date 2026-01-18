from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_service_by_id_or_name(consul_api, service_id_or_name):
    """ iterate the registered services and find one with the given id """
    for dummy, service in consul_api.agent.services().items():
        if service_id_or_name in (service['ID'], service['Service']):
            return ConsulService(loaded=service)