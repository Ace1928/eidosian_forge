from __future__ import absolute_import, division, print_function
import stat
import os
import traceback
from ansible.module_utils.common import respawn
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
@classmethod
def need_root(cls):
    """Check if the module was run as root."""
    if os.geteuid() != 0:
        cls.raise_exception('This command has to be run under the root user.')