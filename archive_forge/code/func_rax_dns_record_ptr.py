from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (rax_argument_spec,
def rax_dns_record_ptr(module, data=None, comment=None, loadbalancer=None, name=None, server=None, state='present', ttl=7200):
    changed = False
    results = []
    dns = pyrax.cloud_dns
    if not dns:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    if loadbalancer:
        item = rax_find_loadbalancer(module, pyrax, loadbalancer)
    elif server:
        item = rax_find_server(module, pyrax, server)
    if state == 'present':
        current = dns.list_ptr_records(item)
        for record in current:
            if record.data == data:
                if record.ttl != ttl or record.name != name:
                    try:
                        dns.update_ptr_record(item, record, name, data, ttl)
                        changed = True
                    except Exception as e:
                        module.fail_json(msg='%s' % e.message)
                    record.ttl = ttl
                    record.name = name
                    results.append(rax_to_dict(record))
                    break
                else:
                    results.append(rax_to_dict(record))
                    break
        if not results:
            record = dict(name=name, type='PTR', data=data, ttl=ttl, comment=comment)
            try:
                results = dns.add_ptr_records(item, [record])
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        module.exit_json(changed=changed, records=results)
    elif state == 'absent':
        current = dns.list_ptr_records(item)
        for record in current:
            if record.data == data:
                results.append(rax_to_dict(record))
                break
        if results:
            try:
                dns.delete_ptr_records(item, data)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        module.exit_json(changed=changed, records=results)