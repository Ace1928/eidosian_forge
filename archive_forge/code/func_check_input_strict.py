from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ..module_utils.podman.podman_container_lib import PodmanManager  # noqa: F402
from ..module_utils.podman.podman_container_lib import set_container_opts  # noqa: F402
def check_input_strict(container):
    if container['state'] in ['started', 'present'] and (not container['image']):
        return "State '%s' required image to be configured!" % container['state']