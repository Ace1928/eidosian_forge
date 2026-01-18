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
def set_mode_if_different(self, path, mode, changed, diff=None, expand=True):
    if mode is None:
        return changed
    b_path = to_bytes(path, errors='surrogate_or_strict')
    if expand:
        b_path = os.path.expanduser(os.path.expandvars(b_path))
    if self.check_file_absent_if_check_mode(b_path):
        return True
    path_stat = os.lstat(b_path)
    if not isinstance(mode, int):
        try:
            mode = int(mode, 8)
        except Exception:
            try:
                mode = self._symbolic_mode_to_octal(path_stat, mode)
            except Exception as e:
                path = to_text(b_path)
                self.fail_json(path=path, msg='mode must be in octal or symbolic form', details=to_native(e))
            if mode != stat.S_IMODE(mode):
                path = to_text(b_path)
                self.fail_json(path=path, msg='Invalid mode supplied, only permission info is allowed', details=mode)
    prev_mode = stat.S_IMODE(path_stat.st_mode)
    if prev_mode != mode:
        if diff is not None:
            if 'before' not in diff:
                diff['before'] = {}
            diff['before']['mode'] = '0%03o' % prev_mode
            if 'after' not in diff:
                diff['after'] = {}
            diff['after']['mode'] = '0%03o' % mode
        if self.check_mode:
            return True
        try:
            if hasattr(os, 'lchmod'):
                os.lchmod(b_path, mode)
            elif not os.path.islink(b_path):
                os.chmod(b_path, mode)
            else:
                underlying_stat = os.stat(b_path)
                os.chmod(b_path, mode)
                new_underlying_stat = os.stat(b_path)
                if underlying_stat.st_mode != new_underlying_stat.st_mode:
                    os.chmod(b_path, stat.S_IMODE(underlying_stat.st_mode))
        except OSError as e:
            if os.path.islink(b_path) and e.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
                pass
            elif e.errno in (errno.ENOENT, errno.ELOOP):
                pass
            else:
                raise
        except Exception as e:
            path = to_text(b_path)
            self.fail_json(path=path, msg='chmod failed', details=to_native(e), exception=traceback.format_exc())
        path_stat = os.lstat(b_path)
        new_mode = stat.S_IMODE(path_stat.st_mode)
        if new_mode != prev_mode:
            changed = True
    return changed