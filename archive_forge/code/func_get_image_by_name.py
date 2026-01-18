from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
def get_image_by_name(module, client, image_name):
    return get_image(module, client, lambda image: image.NAME == image_name)