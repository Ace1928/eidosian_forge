from __future__ import absolute_import, division, print_function
import json
import re
import shlex
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command
def remove_image_id(self, image_id=None):
    if image_id is None:
        image_id = re.sub(':.*$', '', self.image_name)
    args = ['rmi', image_id]
    if self.force:
        args.append('--force')
    rc, out, err = self._run(args, ignore_errors=True)
    if rc != 0:
        self.module.fail_json(msg='Failed to remove image with id {image_id}. {err}'.format(image_id=image_id, err=err))
    return out