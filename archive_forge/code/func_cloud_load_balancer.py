from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (CLB_ALGORITHMS,
def cloud_load_balancer(module, state, name, meta, algorithm, port, protocol, vip_type, timeout, wait, wait_timeout, vip_id):
    if int(timeout) < 30:
        module.fail_json(msg='"timeout" must be greater than or equal to 30')
    changed = False
    balancers = []
    clb = pyrax.cloud_loadbalancers
    if not clb:
        module.fail_json(msg='Failed to instantiate client. This typically indicates an invalid region or an incorrectly capitalized region name.')
    balancer_list = clb.list()
    while balancer_list:
        retrieved = clb.list(marker=balancer_list.pop().id)
        balancer_list.extend(retrieved)
        if len(retrieved) < 2:
            break
    for balancer in balancer_list:
        if name != balancer.name and name != balancer.id:
            continue
        balancers.append(balancer)
    if len(balancers) > 1:
        module.fail_json(msg='Multiple Load Balancers were matched by name, try using the Load Balancer ID instead')
    if state == 'present':
        if isinstance(meta, dict):
            metadata = [dict(key=k, value=v) for k, v in meta.items()]
        if not balancers:
            try:
                virtual_ips = [clb.VirtualIP(type=vip_type, id=vip_id)]
                balancer = clb.create(name, metadata=metadata, port=port, algorithm=algorithm, protocol=protocol, timeout=timeout, virtual_ips=virtual_ips)
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
        else:
            balancer = balancers[0]
            setattr(balancer, 'metadata', [dict(key=k, value=v) for k, v in balancer.get_metadata().items()])
            atts = {'name': name, 'algorithm': algorithm, 'port': port, 'protocol': protocol, 'timeout': timeout}
            for att, value in atts.items():
                current = getattr(balancer, att)
                if current != value:
                    changed = True
            if changed:
                balancer.update(**atts)
            if balancer.metadata != metadata:
                balancer.set_metadata(meta)
                changed = True
            virtual_ips = [clb.VirtualIP(type=vip_type)]
            current_vip_types = set([v.type for v in balancer.virtual_ips])
            vip_types = set([v.type for v in virtual_ips])
            if current_vip_types != vip_types:
                module.fail_json(msg='Load balancer Virtual IP type cannot be changed')
        if wait:
            attempts = wait_timeout // 5
            pyrax.utils.wait_for_build(balancer, interval=5, attempts=attempts)
        balancer.get()
        instance = rax_to_dict(balancer, 'clb')
        result = dict(changed=changed, balancer=instance)
        if balancer.status == 'ERROR':
            result['msg'] = '%s failed to build' % balancer.id
        elif wait and balancer.status not in ('ACTIVE', 'ERROR'):
            result['msg'] = 'Timeout waiting on %s' % balancer.id
        if 'msg' in result:
            module.fail_json(**result)
        else:
            module.exit_json(**result)
    elif state == 'absent':
        if balancers:
            balancer = balancers[0]
            try:
                balancer.delete()
                changed = True
            except Exception as e:
                module.fail_json(msg='%s' % e.message)
            instance = rax_to_dict(balancer, 'clb')
            if wait:
                attempts = wait_timeout // 5
                pyrax.utils.wait_until(balancer, 'status', 'DELETED', interval=5, attempts=attempts)
        else:
            instance = {}
    module.exit_json(changed=changed, balancer=instance)