from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
def clone_image(module, client, image, new_name):
    if new_name is None:
        new_name = 'Copy of ' + image.NAME
    tmp_image = get_image_by_name(module, client, new_name)
    if tmp_image:
        result = get_image_info(tmp_image)
        result['changed'] = False
        return result
    if image.STATE == IMAGE_STATES.index('DISABLED'):
        module.fail_json(msg='Cannot clone DISABLED image')
    if not module.check_mode:
        new_id = client.image.clone(image.ID, new_name)
        wait_for_ready(module, client, new_id)
        image = client.image.info(new_id)
    result = get_image_info(image)
    result['changed'] = True
    return result