from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
def gen_specs(**specs):
    specs.update({'bind_dn': dict(), 'bind_pw': dict(default='', no_log=True), 'ca_path': dict(type='path'), 'dn': dict(required=True), 'referrals_chasing': dict(type='str', default='anonymous', choices=['disabled', 'anonymous']), 'server_uri': dict(default='ldapi:///'), 'start_tls': dict(default=False, type='bool'), 'validate_certs': dict(default=True, type='bool'), 'sasl_class': dict(choices=['external', 'gssapi'], default='external', type='str'), 'xorder_discovery': dict(choices=['enable', 'auto', 'disable'], default='auto', type='str'), 'client_cert': dict(default=None, type='path'), 'client_key': dict(default=None, type='path')})
    return specs