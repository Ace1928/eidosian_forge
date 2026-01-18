from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (rax_argument_spec,
def cloud_load_balancer_ssl(module, loadbalancer, state, enabled, private_key, certificate, intermediate_certificate, secure_port, secure_traffic_only, https_redirect, wait, wait_timeout):
    if state == 'present':
        if not private_key:
            module.fail_json(msg='private_key must be provided.')
        else:
            private_key = private_key.strip()
        if not certificate:
            module.fail_json(msg='certificate must be provided.')
        else:
            certificate = certificate.strip()
    attempts = wait_timeout // 5
    balancer = rax_find_loadbalancer(module, pyrax, loadbalancer)
    existing_ssl = balancer.get_ssl_termination()
    changed = False
    if state == 'present':
        ssl_attrs = dict(securePort=secure_port, privatekey=private_key, certificate=certificate, intermediateCertificate=intermediate_certificate, enabled=enabled, secureTrafficOnly=secure_traffic_only)
        needs_change = False
        if existing_ssl:
            for ssl_attr, value in ssl_attrs.items():
                if ssl_attr == 'privatekey':
                    continue
                if value is not None and existing_ssl.get(ssl_attr) != value:
                    needs_change = True
        else:
            needs_change = True
        if needs_change:
            try:
                balancer.add_ssl_termination(**ssl_attrs)
            except pyrax.exceptions.PyraxException as e:
                module.fail_json(msg='%s' % e.message)
            changed = True
    elif state == 'absent':
        if existing_ssl:
            try:
                balancer.delete_ssl_termination()
            except pyrax.exceptions.PyraxException as e:
                module.fail_json(msg='%s' % e.message)
            changed = True
    if https_redirect is not None and balancer.httpsRedirect != https_redirect:
        if changed:
            pyrax.utils.wait_for_build(balancer, interval=5, attempts=attempts)
        try:
            balancer.update(httpsRedirect=https_redirect)
        except pyrax.exceptions.PyraxException as e:
            module.fail_json(msg='%s' % e.message)
        changed = True
    if changed and wait:
        pyrax.utils.wait_for_build(balancer, interval=5, attempts=attempts)
    balancer.get()
    new_ssl_termination = balancer.get_ssl_termination()
    if new_ssl_termination:
        new_ssl = dict(enabled=new_ssl_termination['enabled'], secure_port=new_ssl_termination['securePort'], secure_traffic_only=new_ssl_termination['secureTrafficOnly'])
    else:
        new_ssl = None
    result = dict(changed=changed, https_redirect=balancer.httpsRedirect, ssl_termination=new_ssl, balancer=rax_to_dict(balancer, 'clb'))
    success = True
    if balancer.status == 'ERROR':
        result['msg'] = '%s failed to build' % balancer.id
        success = False
    elif wait and balancer.status not in ('ACTIVE', 'ERROR'):
        result['msg'] = 'Timeout waiting on %s' % balancer.id
        success = False
    if success:
        module.exit_json(**result)
    else:
        module.fail_json(**result)