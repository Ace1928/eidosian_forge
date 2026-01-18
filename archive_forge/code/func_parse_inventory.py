from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.ciscosmb.plugins.module_utils.ciscosmb import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
def parse_inventory(self, data):
    data = re.sub('\\nPID', '  PID', data)
    data = re.sub('^\\n', '', data)
    data = re.sub('\\n\\n', '', data)
    data = re.sub('\\n\\s*\\n', '\\n', data)
    lines = data.splitlines()
    modules = {}
    for line in lines:
        line = re.sub('"', '', line)
        line = re.sub('\\s+', ' ', line)
        line = re.sub(':\\s', '"', line)
        line = re.sub('\\s+DESCR"', '"DESCR"', line)
        line = re.sub('\\s+PID"', '"PID"', line)
        line = re.sub('\\s+VID"', '"VID"', line)
        line = re.sub('\\s+SN"', '"SN"', line)
        line = re.sub('\\s*$', '', line)
        match = re.search('^NAME"(?P<name>[^"]+)"DESCR"(?P<descr>[^"]+)"PID"(?P<pid>[^"]+)"VID"(?P<vid>[^"]+)"SN"(?P<sn>\\S+)\\s*', line)
        modul = match.groupdict()
        modules[modul['name']] = modul
    if modules:
        return modules