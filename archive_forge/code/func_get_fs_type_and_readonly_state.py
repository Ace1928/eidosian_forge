from __future__ import absolute_import, division, print_function
import os.path
import xml
import re
from xml.dom.minidom import parseString as parseXML
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def get_fs_type_and_readonly_state(mount_point):
    with open('/proc/mounts', 'r') as file:
        for line in file.readlines():
            fields = line.split()
            path = fields[1]
            if path == mount_point:
                fs = fields[2]
                opts = fields[3]
                return (fs, 'ro' in opts.split(','))
    return None