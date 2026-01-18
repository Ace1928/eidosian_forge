from __future__ import (absolute_import, division, print_function)
import abc
import json
import shlex
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.auth import resolve_repository_name
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def find_image_by_id(self, image_id, accept_missing_image=False):
    """
        Lookup an image (by ID) and return the inspection results.
        """
    if not image_id:
        return None
    self.log('Find image %s (by ID)' % image_id)
    rc, image, stderr = self.call_cli_json('image', 'inspect', image_id)
    if not image:
        if not accept_missing_image:
            self.fail('Error inspecting image ID %s - %s' % (image_id, to_native(stderr)))
        self.log('Image %s not found.' % image_id)
        return None
    if rc != 0:
        self.fail('Error inspecting image ID %s - %s' % (image_id, to_native(stderr)))
    return image[0]