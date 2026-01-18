from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_check(module):
    if module.params['check_id'] or any((module.params[p] is not None for p in ('script', 'ttl', 'tcp', 'http'))):
        return ConsulCheck(module.params['check_id'], module.params['check_name'], module.params['check_node'], module.params['check_host'], module.params['script'], module.params['interval'], module.params['ttl'], module.params['notes'], module.params['tcp'], module.params['http'], module.params['timeout'], module.params['service_id'])