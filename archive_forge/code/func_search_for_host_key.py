from __future__ import absolute_import, division, print_function
import base64
import errno
import hashlib
import hmac
import os
import os.path
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def search_for_host_key(module, host, key, path, sshkeygen):
    """search_for_host_key(module,host,key,path,sshkeygen) -> (found,replace_or_add,found_line)

    Looks up host and keytype in the known_hosts file path; if it's there, looks to see
    if one of those entries matches key. Returns:
    found (Boolean): is host found in path?
    replace_or_add (Boolean): is the key in path different to that supplied by user?
    found_line (int or None): the line where a key of the same type was found
    if found=False, then replace is always False.
    sshkeygen is the path to ssh-keygen, found earlier with get_bin_path
    """
    if os.path.exists(path) is False:
        return (False, False, None)
    sshkeygen_command = [sshkeygen, '-F', host, '-f', path]
    rc, stdout, stderr = module.run_command(sshkeygen_command, check_rc=False)
    if stdout == '' and stderr == '' and (rc == 0 or rc == 1):
        return (False, False, None)
    if rc != 0:
        module.fail_json(msg="ssh-keygen failed (rc=%d, stdout='%s',stderr='%s')" % (rc, stdout, stderr))
    if not key:
        return (True, False, None)
    lines = stdout.split('\n')
    new_key = normalize_known_hosts_key(key)
    for lnum, l in enumerate(lines):
        if l == '':
            continue
        elif l[0] == '#':
            try:
                found_line = int(re.search('found: line (\\d+)', l).group(1))
            except IndexError:
                module.fail_json(msg="failed to parse output of ssh-keygen for line number: '%s'" % l)
        else:
            found_key = normalize_known_hosts_key(l)
            if new_key['host'][:3] == '|1|' and found_key['host'][:3] == '|1|':
                new_key['host'] = found_key['host']
            if new_key == found_key:
                return (True, False, found_line)
            elif new_key['type'] == found_key['type']:
                return (True, True, found_line)
    return (True, True, None)