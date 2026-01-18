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
@property
def files_in_archive(self):
    if self._files_in_archive:
        return self._files_in_archive
    cmd = [self.cmd_path, '--list', '-C', self.b_dest]
    if self.zipflag:
        cmd.append(self.zipflag)
    if self.opts:
        cmd.extend(['--show-transformed-names'] + self.opts)
    if self.excludes:
        cmd.extend(['--exclude=' + f for f in self.excludes])
    cmd.extend(['-f', self.src])
    if self.include_files:
        cmd.extend(self.include_files)
    locale = get_best_parsable_locale(self.module)
    rc, out, err = self.module.run_command(cmd, cwd=self.b_dest, environ_update=dict(LANG=locale, LC_ALL=locale, LC_MESSAGES=locale, LANGUAGE=locale))
    if rc != 0:
        self.module.debug(err)
        raise UnarchiveError('Unable to list files in the archive: %s' % err)
    for filename in out.splitlines():
        filename = to_native(codecs.escape_decode(filename)[0])
        if filename.startswith('/'):
            filename = filename[1:]
        exclude_flag = False
        if self.excludes:
            for exclude in self.excludes:
                if fnmatch.fnmatch(filename, exclude):
                    exclude_flag = True
                    break
        if not exclude_flag:
            self._files_in_archive.append(to_native(filename))
    return self._files_in_archive