import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def save_host_keys():
    """
    Save "discovered" host keys in $(config)/ssh_host_keys/.
    """
    global SYSTEM_HOSTKEYS, BRZ_HOSTKEYS
    bzr_hostkey_path = _ssh_host_keys_config_dir()
    bedding.ensure_config_dir_exists()
    try:
        with open(bzr_hostkey_path, 'w') as f:
            f.write('# SSH host keys collected by bzr\n')
            for hostname, keys in BRZ_HOSTKEYS.items():
                for keytype, key in keys.items():
                    f.write('%s %s %s\n' % (hostname, keytype, key.get_base64()))
    except OSError as e:
        trace.mutter('failed to save bzr host keys: ' + str(e))