from __future__ import absolute_import, division, print_function
import errno
import os
import shutil
import sys
import time
from pwd import getpwnam, getpwuid
from grp import getgrnam, getgrgid
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def get_timestamp_for_time(formatted_time, time_format):
    if formatted_time == 'preserve':
        return None
    elif formatted_time == 'now':
        return Sentinel
    else:
        try:
            struct = time.strptime(formatted_time, time_format)
            struct_time = time.mktime(struct)
        except (ValueError, OverflowError) as e:
            raise AnsibleModuleError(results={'msg': 'Error while obtaining timestamp for time %s using format %s: %s' % (formatted_time, time_format, to_native(e, nonstring='simplerepr'))})
        return struct_time