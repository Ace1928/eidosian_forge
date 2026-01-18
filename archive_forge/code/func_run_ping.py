from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def run_ping(module, blade):
    """Run network ping"""
    ping_fact = {}
    if module.params['source'] and module.params['component']:
        res = blade.get_network_interfaces_ping(destination=module.params['destination'], component=module.params['component'], source=module.params['source'], packet_size=module.params['packet_size'], count=module.params['count'], print_latency=module.params['latency'], resolve_hostname=module.params['resolve'])
    elif module.params['source'] and (not module.params['component']):
        res = blade.get_network_interfaces_ping(destination=module.params['destination'], source=module.params['source'], packet_size=module.params['packet_size'], count=module.params['count'], print_latency=module.params['latency'], resolve_hostname=module.params['resolve'])
    elif not module.params['source'] and module.params['component']:
        res = blade.get_network_interfaces_ping(destination=module.params['destination'], component=module.params['component'], packet_size=module.params['packet_size'], count=module.params['count'], print_latency=module.params['latency'], resolve_hostname=module.params['resolve'])
    else:
        res = blade.get_network_interfaces_ping(destination=module.params['destination'], packet_size=module.params['packet_size'], count=module.params['count'], print_latency=module.params['latency'], resolve_hostname=module.params['resolve'])
    if res.status_code != 200:
        module.fail_json(msg='Failed to run ping. Error: {0}'.format(res.errors[0].message))
    else:
        responses = list(res.items)
        for resp in range(0, len(responses)):
            comp_name = responses[resp].component_name.replace('.', '_')
            ping_fact[comp_name] = {'details': responses[resp].details}
    module.exit_json(changed=False, pingfact=ping_fact)