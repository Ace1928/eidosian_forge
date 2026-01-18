from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def get_ssh_key_fingerprint(ssh_key, hash_algo='sha256'):
    """
    Return the public key fingerprint of a given public SSH key
    in format "[fp] [comment] (ssh-rsa)" where fp is of the format:
    FB:0C:AC:0A:07:94:5B:CE:75:6E:63:32:13:AD:AD:D7
    for md5 or
    SHA256:[base64]
    for sha256
    Comments are assumed to be all characters past the second
    whitespace character in the sshpubkey string.
    :param ssh_key:
    :param hash_algo:
    :return:
    """
    parts = ssh_key.strip().split(None, 2)
    if len(parts) == 0:
        return None
    key_type = parts[0]
    key = base64.b64decode(parts[1].encode('ascii'))
    if hash_algo == 'md5':
        fp_plain = hashlib.md5(key).hexdigest()
        key_fp = ':'.join((a + b for a, b in zip(fp_plain[::2], fp_plain[1::2]))).upper()
    elif hash_algo == 'sha256':
        fp_plain = base64.b64encode(hashlib.sha256(key).digest()).decode('ascii').rstrip('=')
        key_fp = 'SHA256:{fp}'.format(fp=fp_plain)
    if len(parts) < 3:
        return '%s (%s)' % (key_fp, key_type)
    else:
        comment = parts[2]
        return '%s %s (%s)' % (key_fp, comment, key_type)