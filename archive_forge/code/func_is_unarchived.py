from __future__ import absolute_import, division, print_function
import binascii
import codecs
import datetime
import fnmatch
import grp
import os
import platform
import pwd
import re
import stat
import time
import traceback
from functools import partial
from zipfile import ZipFile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_file
def is_unarchived(self):
    cmd = [self.cmd_path, '--diff', '-C', self.b_dest]
    if self.zipflag:
        cmd.append(self.zipflag)
    if self.opts:
        cmd.extend(['--show-transformed-names'] + self.opts)
    if self.file_args['owner']:
        cmd.append('--owner=' + quote(self.file_args['owner']))
    if self.file_args['group']:
        cmd.append('--group=' + quote(self.file_args['group']))
    if self.module.params['keep_newer']:
        cmd.append('--keep-newer-files')
    if self.excludes:
        cmd.extend(['--exclude=' + f for f in self.excludes])
    cmd.extend(['-f', self.src])
    if self.include_files:
        cmd.extend(self.include_files)
    locale = get_best_parsable_locale(self.module)
    rc, out, err = self.module.run_command(cmd, cwd=self.b_dest, environ_update=dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LANGUAGE=locale))
    unarchived = True
    old_out = out
    out = ''
    run_uid = os.getuid()
    for line in old_out.splitlines() + err.splitlines():
        if EMPTY_FILE_RE.search(line):
            continue
        if run_uid == 0 and (not self.file_args['owner']) and OWNER_DIFF_RE.search(line):
            out += line + '\n'
        if run_uid == 0 and (not self.file_args['group']) and GROUP_DIFF_RE.search(line):
            out += line + '\n'
        if not self.file_args['mode'] and MODE_DIFF_RE.search(line):
            out += line + '\n'
        if MOD_TIME_DIFF_RE.search(line):
            out += line + '\n'
        if MISSING_FILE_RE.search(line):
            out += line + '\n'
        if INVALID_OWNER_RE.search(line):
            out += line + '\n'
        if INVALID_GROUP_RE.search(line):
            out += line + '\n'
    if out:
        unarchived = False
    return dict(unarchived=unarchived, rc=rc, out=out, err=err, cmd=cmd)