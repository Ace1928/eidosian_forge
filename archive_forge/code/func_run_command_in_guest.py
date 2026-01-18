from __future__ import absolute_import, division, print_function
import atexit
import ansible.module_utils.common._collections_compat as collections_compat
import json
import os
import re
import socket
import ssl
import hashlib
import time
import traceback
import datetime
from collections import OrderedDict
from ansible.module_utils.compat.version import StrictVersion
from random import randint
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.six import integer_types, iteritems, string_types, raise_from
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import unquote
def run_command_in_guest(content, vm, username, password, program_path, program_args, program_cwd, program_env):
    result = {'failed': False}
    tools_status = vm.guest.toolsStatus
    if tools_status == 'toolsNotInstalled' or tools_status == 'toolsNotRunning':
        result['failed'] = True
        result['msg'] = 'VMwareTools is not installed or is not running in the guest'
        return result
    creds = vim.vm.guest.NamePasswordAuthentication(username=username, password=password)
    try:
        pm = content.guestOperationsManager.processManager
        ps = vim.vm.guest.ProcessManager.ProgramSpec(programPath=program_path, arguments=program_args, workingDirectory=program_cwd)
        res = pm.StartProgramInGuest(vm, creds, ps)
        result['pid'] = res
        pdata = pm.ListProcessesInGuest(vm, creds, [res])
        while not pdata[0].endTime:
            time.sleep(1)
            pdata = pm.ListProcessesInGuest(vm, creds, [res])
        result['owner'] = pdata[0].owner
        result['startTime'] = pdata[0].startTime.isoformat()
        result['endTime'] = pdata[0].endTime.isoformat()
        result['exitCode'] = pdata[0].exitCode
        if result['exitCode'] != 0:
            result['failed'] = True
            result['msg'] = 'program exited non-zero'
        else:
            result['msg'] = 'program completed successfully'
    except Exception as e:
        result['msg'] = str(e)
        result['failed'] = True
    return result