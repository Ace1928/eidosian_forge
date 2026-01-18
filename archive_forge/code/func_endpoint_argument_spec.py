from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def endpoint_argument_spec():
    return dict(role=dict(), hostname=dict(required=True), port=dict(type='int'), validate_certs=dict(default=True, type='bool', aliases=['verify_ssl']), certificate_authority=dict(), security_protocol=dict(choices=['ssl-with-validation', 'ssl-with-validation-custom-ca', 'ssl-without-validation', 'non-ssl']), userid=dict(), password=dict(no_log=True), auth_key=dict(no_log=True), subscription=dict(no_log=True), project=dict(), uid_ems=dict(), path=dict())