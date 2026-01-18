from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import (
def format_commands(commands):
    """
    This function format the input commands and removes the prepend white spaces
    for command lines having 'set' or 'delete' and it skips empty lines.
    :param commands:
    :return: list of commands
    """
    return [line.strip() if line.split()[0] in ('set', 'delete') else line for line in commands if len(line.strip()) > 0]