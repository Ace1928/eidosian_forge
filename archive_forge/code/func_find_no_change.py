from __future__ import absolute_import, division, print_function
import copy
import re
import shlex
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Conditional
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from collections import deque
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def find_no_change(self, responses):
    """Searches the response for something that looks like a change

        This method borrows heavily from Ansible's ``_find_prompt`` method
        defined in the ``lib/ansible/plugins/connection/network_cli.py::Connection``
        class.

        Arguments:
            response (string): The output from the command.

        Returns:
            bool: True when change is detected. False otherwise.
        """
    for response in responses:
        for regex in self.stdout_re:
            if regex.search(response):
                return True
    return False