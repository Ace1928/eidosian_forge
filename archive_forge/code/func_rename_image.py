from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
def rename_image(module, client, image, new_name):
    if new_name is None:
        module.fail_json(msg="'new_name' option has to be specified when the state is 'renamed'")
    if new_name == image.NAME:
        result = get_image_info(image)
        result['changed'] = False
        return result
    tmp_image = get_image_by_name(module, client, new_name)
    if tmp_image:
        module.fail_json(msg="Name '" + new_name + "' is already taken by IMAGE with id=" + str(tmp_image.ID))
    if not module.check_mode:
        client.image.rename(image.ID, new_name)
    result = get_image_info(image)
    result['changed'] = True
    return result