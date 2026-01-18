from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_find_volume(module, rax_module, name):
    """Find a Block storage volume by ID or name"""
    cbs = rax_module.cloud_blockstorage
    try:
        UUID(name)
        volume = cbs.get(name)
    except ValueError:
        try:
            volume = cbs.find(name=name)
        except rax_module.exc.NotFound:
            volume = None
        except Exception as e:
            module.fail_json(msg='%s' % e)
    return volume