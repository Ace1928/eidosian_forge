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
def pick_handler(src, dest, file_args, module):
    handlers = [ZipArchive, ZipZArchive, TgzArchive, TarArchive, TarBzipArchive, TarXzArchive, TarZstdArchive]
    reasons = set()
    for handler in handlers:
        obj = handler(src, dest, file_args, module)
        can_handle, reason = obj.can_handle_archive()
        if can_handle:
            return obj
        reasons.add(reason)
    reason_msg = '\n'.join(reasons)
    module.fail_json(msg='Failed to find handler for "%s". Make sure the required command to extract the file is installed.\n%s' % (src, reason_msg))