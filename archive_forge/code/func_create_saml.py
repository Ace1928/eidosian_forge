from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def create_saml(module, array):
    """Create SAML2 IdP"""
    changed = True
    if not module.check_mode:
        sp = Saml2SsoSp(decryption_credential=ReferenceNoId(name=module.params['decryption_credential']), signing_credential=ReferenceNoId(name=module.params['signing_credential']))
        idp = Saml2SsoIdp(url=module.params['url'], metadata_url=module.params['metadata_url'], sign_request_enabled=module.params['sign_request'], encrypt_assertion_enabled=module.params['encrypt_asserts'], verification_certificate=module.params['x509_cert'])
        if not module.check_mode:
            res = array.post_sso_saml2_idps(idp=Saml2SsoPost(array_url=module.params['array_url'], idp=idp, sp=sp), names=[module.params['name']])
            if res.status_code != 200:
                module.fail_json(msg='Failed to create SAML2 Identity Provider {0}. Error message: {1}'.format(module.params['name'], res.errors[0].message))
            if module.params['enabled']:
                res = array.patch_sso_saml2_idps(idp=Saml2Sso(enabled=module.params['enabled']), names=[module.params['name']])
                if res.status_code != 200:
                    array.delete_sso_saml2_idps(names=[module.params['name']])
                    module.fail_json(msg='Failed to create SAML2 Identity Provider {0}. Error message: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)