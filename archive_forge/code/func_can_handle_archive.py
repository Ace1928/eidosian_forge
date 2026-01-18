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
def can_handle_archive(self):
    unzip_available, error_msg = super(ZipZArchive, self).can_handle_archive()
    if not unzip_available:
        return (unzip_available, error_msg)
    cmd = [self.zipinfo_cmd_path, self.zipinfoflag]
    rc, out, err = self.module.run_command(cmd)
    if 'zipinfo' in out.lower():
        return (True, None)
    return (False, 'Command "unzip -Z" could not handle archive: %s' % err)