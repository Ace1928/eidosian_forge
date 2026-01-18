from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
def ensure_image_exists(module, image, module_params):
    """If image is passed, ensure it exists, if not - pull it or fail.

    Arguments:
        module {obj} -- ansible module object
        image {str} -- name of image

    Returns:
        list -- list of image actions - if it pulled or nothing was done
    """
    image_actions = []
    module_exec = module_params['executable']
    is_rootfs = module_params['rootfs']
    if is_rootfs:
        if not os.path.exists(image) or not os.path.isdir(image):
            module.fail_json(msg="Image rootfs doesn't exist %s" % image)
        return image_actions
    if not image:
        return image_actions
    rc, out, err = module.run_command([module_exec, 'image', 'exists', image])
    if rc == 0:
        return image_actions
    rc, out, err = module.run_command([module_exec, 'image', 'pull', image])
    if rc != 0:
        module.fail_json(msg="Can't pull image %s" % image, stdout=out, stderr=err)
    image_actions.append('pulled image %s' % image)
    return image_actions