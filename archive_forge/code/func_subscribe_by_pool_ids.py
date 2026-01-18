from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def subscribe_by_pool_ids(self, pool_ids):
    """
        Try to subscribe to the list of pool IDs
        """
    available_pools = RhsmPools(self.module)
    available_pool_ids = [p.get_pool_id() for p in available_pools]
    for pool_id, quantity in sorted(pool_ids.items()):
        if pool_id in available_pool_ids:
            args = [SUBMAN_CMD, 'attach', '--pool', pool_id]
            if quantity is not None:
                args.extend(['--quantity', to_native(quantity)])
            rc, stderr, stdout = self.module.run_command(args, check_rc=True)
        else:
            self.module.fail_json(msg='Pool ID: %s not in list of available pools' % pool_id)
    return pool_ids