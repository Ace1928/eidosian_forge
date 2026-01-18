from __future__ import absolute_import, division, print_function
import json
import os
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.rax import (FINAL_STATUSES, rax_argument_spec, rax_find_bootable_volume,
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.six import string_types
def rax_find_server_image(module, server, image, boot_volume):
    if not image and boot_volume:
        vol = rax_find_bootable_volume(module, pyrax, server, exit=False)
        if not vol:
            return None
        volume_image_metadata = vol.volume_image_metadata
        vol_image_id = volume_image_metadata.get('image_id')
        if vol_image_id:
            server_image = rax_find_image(module, pyrax, vol_image_id, exit=False)
            if server_image:
                server.image = dict(id=server_image)
    if image and (not server.image):
        vol = rax_find_bootable_volume(module, pyrax, server)
        volume_image_metadata = vol.volume_image_metadata
        vol_image_id = volume_image_metadata.get('image_id')
        if not vol_image_id:
            return None
        server_image = rax_find_image(module, pyrax, vol_image_id, exit=False)
        if image != server_image:
            return None
        server.image = dict(id=server_image)
    elif image and server.image['id'] != image:
        return None
    return server.image