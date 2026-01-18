from __future__ import absolute_import, division, print_function
import os
import platform
import pwd
import re
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.six.moves import shlex_quote
def find_job(self, name, job=None):
    comment = None
    for l in self.lines:
        if comment is not None:
            if comment == name:
                return [comment, l]
            else:
                comment = None
        elif re.match('%s' % self.ansible, l):
            comment = re.sub('%s' % self.ansible, '', l)
    if job:
        for i, l in enumerate(self.lines):
            if l == job:
                if not re.match('%s' % self.ansible, self.lines[i - 1]):
                    self.lines.insert(i, self.do_comment(name))
                    return [self.lines[i], l, True]
                elif name and self.lines[i - 1] == self.do_comment(None):
                    self.lines[i - 1] = self.do_comment(name)
                    return [self.lines[i - 1], l, True]
    return []