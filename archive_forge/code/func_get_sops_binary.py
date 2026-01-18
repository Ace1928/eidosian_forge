from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
@staticmethod
def get_sops_binary(get_option_value):
    cmd = get_option_value('sops_binary') if get_option_value else None
    if cmd is None:
        cmd = 'sops'
    return cmd