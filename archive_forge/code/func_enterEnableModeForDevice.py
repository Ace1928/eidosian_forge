from __future__ import absolute_import, division, print_function
import time
import socket
import re
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, EntityCollection
from ansible.module_utils.connection import Connection, exec_command
from ansible.module_utils.connection import ConnectionError
def enterEnableModeForDevice(enablePassword, timeout, obj):
    command = 'enable\n'
    pwdPrompt = 'password:'
    obj.settimeout(int(timeout))
    obj.send(command)
    flag = False
    retVal = ''
    count = 5
    while not flag:
        if count == 0:
            flag = True
        else:
            count = count - 1
        time.sleep(1)
        try:
            buffByte = obj.recv(9999)
            buff = buffByte.decode()
            retVal = retVal + buff
            gotit = buff.find(pwdPrompt)
            if gotit != -1:
                time.sleep(1)
                if enablePassword is None or enablePassword == '':
                    return '\n Error-106'
                obj.send(enablePassword)
                obj.send('\r')
                obj.send('\n')
                time.sleep(1)
                innerBuffByte = obj.recv(9999)
                innerBuff = innerBuffByte.decode()
                retVal = retVal + innerBuff
                innerGotit = innerBuff.find('#')
                if innerGotit != -1:
                    return retVal
            else:
                gotit = buff.find('#')
                if gotit != -1:
                    return retVal
        except Exception:
            retVal = retVal + '\n Error-101'
            flag = True
    if retVal == '':
        retVal = '\n Error-101'
    return retVal