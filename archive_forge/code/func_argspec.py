from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
@property
def argspec(self):
    spec = copy.deepcopy(AUTH_ARG_SPEC)
    spec.update(copy.deepcopy(WAIT_ARG_SPEC))
    spec.update(copy.deepcopy(COMMON_ARG_SPEC))
    spec['service'] = dict(type='str', aliases=['svc'])
    spec['namespace'] = dict(required=True, type='str')
    spec['labels'] = dict(type='dict')
    spec['name'] = dict(type='str')
    spec['hostname'] = dict(type='str')
    spec['path'] = dict(type='str')
    spec['wildcard_policy'] = dict(choices=['Subdomain'], type='str')
    spec['port'] = dict(type='str')
    spec['tls'] = dict(type='dict', options=dict(ca_certificate=dict(type='str'), certificate=dict(type='str'), destination_ca_certificate=dict(type='str'), key=dict(type='str', no_log=False), insecure_policy=dict(type='str', choices=['allow', 'redirect', 'disallow'], default='disallow')))
    spec['termination'] = dict(choices=['edge', 'passthrough', 'reencrypt', 'insecure'], default='insecure')
    spec['annotations'] = dict(type='dict')
    return spec