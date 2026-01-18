from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def define_argument_spec():
    """
        Define the argument spec for the ansible module
        :return: argument spec dictionary
        """
    argument_spec = dict(name=dict(required=True), description=dict(), location=dict(required=True), alias=dict(required=True), port=dict(choices=[80, 443]), method=dict(choices=['leastConnection', 'roundRobin']), persistence=dict(choices=['standard', 'sticky']), nodes=dict(type='list', default=[], elements='dict'), status=dict(default='enabled', choices=['enabled', 'disabled']), state=dict(default='present', choices=['present', 'absent', 'port_absent', 'nodes_present', 'nodes_absent']))
    return argument_spec