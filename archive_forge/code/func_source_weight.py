from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def source_weight(self):
    """Report the weight of this source net.

        Basically this is the netmask, where IPv4 is normalized to IPv6
        (IPv4/32 has the same weight as IPv6/128).
        """
    if self['type'] == 'local':
        return 130
    sourceobj = self.source()
    if isinstance(sourceobj, ipaddress.IPv4Network):
        return sourceobj.prefixlen * 4
    if isinstance(sourceobj, ipaddress.IPv6Network):
        return sourceobj.prefixlen
    if isinstance(sourceobj, str):
        if sourceobj == 'all':
            return 0
        if sourceobj == 'samehost':
            return 129
        if sourceobj == 'samenet':
            return 96
        if sourceobj[0] == '.':
            return 64
        return 128
    raise PgHbaValueError('Cannot deduct the source weight of this source {sourceobj}'.format(sourceobj=sourceobj))