from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
@xcli_wrapper
def execute_pyxcli_command(module, xcli_command, xcli_client):
    pyxcli_args = build_pyxcli_command(module.params)
    getattr(xcli_client.cmd, xcli_command)(**pyxcli_args)
    return True