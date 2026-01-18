from __future__ import absolute_import, division, print_function
import os.path
import xml
import re
from xml.dom.minidom import parseString as parseXML
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def repo_refresh(m):
    """update the repositories"""
    retvals = {'rc': 0, 'stdout': '', 'stderr': ''}
    cmd = get_cmd(m, 'refresh')
    retvals['cmd'] = cmd
    result, retvals['rc'], retvals['stdout'], retvals['stderr'] = parse_zypper_xml(m, cmd)
    return retvals