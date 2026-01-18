from __future__ import absolute_import, division, print_function
import re
from ansible.errors import AnsibleLookupError
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def etcd3_client(client_params):
    try:
        etcd = etcd3.client(**client_params)
        etcd.status()
    except Exception as exp:
        raise AnsibleLookupError('Cannot connect to etcd cluster: %s' % to_native(exp))
    return etcd