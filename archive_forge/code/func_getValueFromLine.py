from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
def getValueFromLine(s):
    spaceRe = re.compile('\\s+')
    m = list(spaceRe.finditer(s))[-1]
    valueEnd = m.start()
    option = s.split()[0]
    optionStart = s.find(option)
    optionLen = len(option)
    return s[optionLen + optionStart:].strip()