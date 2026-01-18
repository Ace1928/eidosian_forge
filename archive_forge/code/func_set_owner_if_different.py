from __future__ import absolute_import, division, print_function
import sys
import __main__
import atexit
import errno
import datetime
import grp
import fcntl
import locale
import os
import pwd
import platform
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import traceback
import types
from itertools import chain, repeat
from ansible.module_utils.compat import selectors
from ._text import to_native, to_bytes, to_text
from ansible.module_utils.common.text.converters import (
from ansible.module_utils.common.arg_spec import ModuleArgumentSpecValidator
from ansible.module_utils.common.text.formatters import (
import hashlib
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.file import (
from ansible.module_utils.common.sys_info import (
from ansible.module_utils.pycompat24 import get_exception, literal_eval
from ansible.module_utils.common.parameters import (
from ansible.module_utils.errors import AnsibleFallbackNotFound, AnsibleValidationErrorMultiple, UnsupportedError
from ansible.module_utils.six import (
from ansible.module_utils.six.moves import map, reduce, shlex_quote
from ansible.module_utils.common.validation import (
from ansible.module_utils.common._utils import get_all_subclasses as _get_all_subclasses
from ansible.module_utils.parsing.convert_bool import BOOLEANS, BOOLEANS_FALSE, BOOLEANS_TRUE, boolean
from ansible.module_utils.common.warnings import (
def set_owner_if_different(self, path, owner, changed, diff=None, expand=True):
    if owner is None:
        return changed
    b_path = to_bytes(path, errors='surrogate_or_strict')
    if expand:
        b_path = os.path.expanduser(os.path.expandvars(b_path))
    if self.check_file_absent_if_check_mode(b_path):
        return True
    orig_uid, orig_gid = self.user_and_group(b_path, expand)
    try:
        uid = int(owner)
    except ValueError:
        try:
            uid = pwd.getpwnam(owner).pw_uid
        except KeyError:
            path = to_text(b_path)
            self.fail_json(path=path, msg='chown failed: failed to look up user %s' % owner)
    if orig_uid != uid:
        if diff is not None:
            if 'before' not in diff:
                diff['before'] = {}
            diff['before']['owner'] = orig_uid
            if 'after' not in diff:
                diff['after'] = {}
            diff['after']['owner'] = uid
        if self.check_mode:
            return True
        try:
            os.lchown(b_path, uid, -1)
        except (IOError, OSError) as e:
            path = to_text(b_path)
            self.fail_json(path=path, msg='chown failed: %s' % to_text(e))
        changed = True
    return changed