from __future__ import absolute_import, division, print_function
import os
import pwd
import os.path
import tempfile
import re
import shlex
from operator import itemgetter
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def parsekey(module, raw_key, rank=None):
    """
    parses a key, which may or may not contain a list
    of ssh-key options at the beginning

    rank indicates the keys original ordering, so that
    it can be written out in the same order.
    """
    VALID_SSH2_KEY_TYPES = ['sk-ecdsa-sha2-nistp256@openssh.com', 'sk-ecdsa-sha2-nistp256-cert-v01@openssh.com', 'webauthn-sk-ecdsa-sha2-nistp256@openssh.com', 'ecdsa-sha2-nistp256', 'ecdsa-sha2-nistp256-cert-v01@openssh.com', 'ecdsa-sha2-nistp384', 'ecdsa-sha2-nistp384-cert-v01@openssh.com', 'ecdsa-sha2-nistp521', 'ecdsa-sha2-nistp521-cert-v01@openssh.com', 'sk-ssh-ed25519@openssh.com', 'sk-ssh-ed25519-cert-v01@openssh.com', 'ssh-ed25519', 'ssh-ed25519-cert-v01@openssh.com', 'ssh-dss', 'ssh-rsa', 'ssh-xmss@openssh.com', 'ssh-xmss-cert-v01@openssh.com', 'rsa-sha2-256', 'rsa-sha2-512', 'ssh-rsa-cert-v01@openssh.com', 'rsa-sha2-256-cert-v01@openssh.com', 'rsa-sha2-512-cert-v01@openssh.com', 'ssh-dss-cert-v01@openssh.com']
    options = None
    key = None
    key_type = None
    type_index = None
    raw_key = raw_key.replace('\\#', '#')
    lex = shlex.shlex(raw_key)
    lex.quotes = []
    lex.commenters = ''
    lex.whitespace_split = True
    key_parts = list(lex)
    if key_parts and key_parts[0] == '#':
        return (raw_key, 'skipped', None, None, rank)
    for i in range(0, len(key_parts)):
        if key_parts[i] in VALID_SSH2_KEY_TYPES:
            type_index = i
            key_type = key_parts[i]
            break
    if type_index is None:
        return None
    elif type_index > 0:
        options = ' '.join(key_parts[:type_index])
    options = parseoptions(module, options)
    key = key_parts[type_index + 1]
    if len(key_parts) > type_index + 1:
        comment = ' '.join(key_parts[type_index + 2:])
    return (key, key_type, options, comment, rank)