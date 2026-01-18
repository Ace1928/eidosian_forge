from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.netscaler.netscaler import (
def default_lb_vserver_identical(client, module):
    d = get_default_lb_vserver(client, module)
    configured = ConfigProxy(actual=csvserver_lbvserver_binding(), client=client, readwrite_attrs=['name', 'lbvserver'], attribute_values_dict={'name': module.params['name'], 'lbvserver': module.params['lbvserver']})
    log('default lb vserver %s' % ((d.name, d.lbvserver),))
    if d.name is None and module.params['lbvserver'] is None:
        log('Default lb vserver identical missing')
        return True
    elif d.name is not None and module.params['lbvserver'] is None:
        log('Default lb vserver needs removing')
        return False
    elif configured.has_equal_attributes(d):
        log('Default lb vserver identical')
        return True
    else:
        log('Default lb vserver not identical')
        return False