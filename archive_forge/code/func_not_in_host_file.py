from __future__ import (absolute_import, division, print_function)
import os
import hmac
import re
from ansible.module_utils.six.moves.urllib.parse import urlparse
def not_in_host_file(self, host):
    if 'USER' in os.environ:
        user_host_file = os.path.expandvars('~${USER}/.ssh/known_hosts')
    else:
        user_host_file = '~/.ssh/known_hosts'
    user_host_file = os.path.expanduser(user_host_file)
    host_file_list = [user_host_file, '/etc/ssh/ssh_known_hosts', '/etc/ssh/ssh_known_hosts2', '/etc/openssh/ssh_known_hosts']
    hfiles_not_found = 0
    for hf in host_file_list:
        if not os.path.exists(hf):
            hfiles_not_found += 1
            continue
        try:
            host_fh = open(hf)
        except IOError:
            hfiles_not_found += 1
            continue
        else:
            data = host_fh.read()
            host_fh.close()
        for line in data.split('\n'):
            if line is None or ' ' not in line:
                continue
            tokens = line.split()
            if tokens[0].find(HASHED_KEY_MAGIC) == 0:
                try:
                    kn_salt, kn_host = tokens[0][len(HASHED_KEY_MAGIC):].split('|', 2)
                    hash = hmac.new(kn_salt.decode('base64'), digestmod=sha1)
                    hash.update(host)
                    if hash.digest() == kn_host.decode('base64'):
                        return False
                except Exception:
                    continue
            elif host in tokens[0]:
                return False
    return True