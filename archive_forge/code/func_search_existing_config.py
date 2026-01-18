from __future__ import absolute_import, division, print_function
import os
import time
from ansible.module_utils.basic import AnsibleModule
def search_existing_config(config, option):
    """ search in config file for specified option """
    if config and os.path.isfile(config):
        with open(config, 'r') as f:
            for line in f:
                if option in line:
                    return line
    return None