from __future__ import absolute_import, division, print_function
import errno
import fnmatch
import grp
import os
import pwd
import re
import stat
import time
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
def contentfilter(fsname, pattern, read_whole_file=False):
    """
    Filter files which contain the given expression
    :arg fsname: Filename to scan for lines matching a pattern
    :arg pattern: Pattern to look for inside of line
    :arg read_whole_file: If true, the whole file is read into memory before the regex is applied against it. Otherwise, the regex is applied line-by-line.
    :rtype: bool
    :returns: True if one of the lines in fsname matches the pattern. Otherwise False
    """
    if pattern is None:
        return True
    prog = re.compile(pattern)
    try:
        with open(fsname) as f:
            if read_whole_file:
                return bool(prog.search(f.read()))
            for line in f:
                if prog.match(line):
                    return True
    except Exception:
        pass
    return False