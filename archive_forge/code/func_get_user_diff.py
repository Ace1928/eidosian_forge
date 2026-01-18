from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_user_diff(client, ipa_user, module_user):
    """
        Return the keys of each dict whereas values are different. Unfortunately the IPA
        API returns everything as a list even if only a single value is possible.
        Therefore some more complexity is needed.
        The method will check if the value type of module_user.attr is not a list and
        create a list with that element if the same attribute in ipa_user is list. In this way I hope that the method
        must not be changed if the returned API dict is changed.
    :param ipa_user:
    :param module_user:
    :return:
    """
    sshpubkey = None
    if 'ipasshpubkey' in module_user:
        hash_algo = 'md5'
        if 'sshpubkeyfp' in ipa_user and ipa_user['sshpubkeyfp'][0][:7].upper() == 'SHA256:':
            hash_algo = 'sha256'
        module_user['sshpubkeyfp'] = [get_ssh_key_fingerprint(pubkey, hash_algo) for pubkey in module_user['ipasshpubkey']]
        sshpubkey = module_user['ipasshpubkey']
        del module_user['ipasshpubkey']
    result = client.get_diff(ipa_data=ipa_user, module_data=module_user)
    if sshpubkey is not None:
        del module_user['sshpubkeyfp']
        module_user['ipasshpubkey'] = sshpubkey
    return result