from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_find_bootable_volume(module, rax_module, server, exit=True):
    """Find a servers bootable volume"""
    cs = rax_module.cloudservers
    cbs = rax_module.cloud_blockstorage
    server_id = rax_module.utils.get_id(server)
    volumes = cs.volumes.get_server_volumes(server_id)
    bootable_volumes = []
    for volume in volumes:
        vol = cbs.get(volume)
        if module.boolean(vol.bootable):
            bootable_volumes.append(vol)
    if not bootable_volumes:
        if exit:
            module.fail_json(msg='No bootable volumes could be found for server %s' % server_id)
        else:
            return False
    elif len(bootable_volumes) > 1:
        if exit:
            module.fail_json(msg='Multiple bootable volumes found for server %s' % server_id)
        else:
            return False
    return bootable_volumes[0]