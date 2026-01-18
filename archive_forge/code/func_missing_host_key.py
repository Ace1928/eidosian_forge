from __future__ import (annotations, absolute_import, division, print_function)
import os
import socket
import tempfile
import traceback
import fcntl
import re
import typing as t
from ansible.module_utils.compat.version import LooseVersion
from binascii import hexlify
from ansible.errors import (
from ansible.module_utils.compat.paramiko import PARAMIKO_IMPORT_ERR, paramiko
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def missing_host_key(self, client, hostname, key) -> None:
    if all((self.connection.get_option('host_key_checking'), not self.connection.get_option('host_key_auto_add'))):
        fingerprint = hexlify(key.get_fingerprint())
        ktype = key.get_name()
        if self.connection.get_option('use_persistent_connections') or self.connection.force_persistence:
            raise AnsibleError(AUTHENTICITY_MSG[1:92] % (hostname, ktype, fingerprint))
        inp = to_text(display.prompt_until(AUTHENTICITY_MSG % (hostname, ktype, fingerprint), private=False), errors='surrogate_or_strict')
        if inp not in ['yes', 'y', '']:
            raise AnsibleError('host connection rejected by user')
    key._added_by_ansible_this_time = True
    client._host_keys.add(hostname, key.get_name(), key)