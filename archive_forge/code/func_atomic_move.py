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
def atomic_move(self, src, dest, unsafe_writes=False):
    """atomically move src to dest, copying attributes from dest, returns true on success
        it uses os.rename to ensure this as it is an atomic operation, rest of the function is
        to work around limitations, corner cases and ensure selinux context is saved if possible"""
    context = None
    dest_stat = None
    b_src = to_bytes(src, errors='surrogate_or_strict')
    b_dest = to_bytes(dest, errors='surrogate_or_strict')
    if os.path.exists(b_dest):
        try:
            dest_stat = os.stat(b_dest)
            os.chmod(b_src, dest_stat.st_mode & PERM_BITS)
            os.chown(b_src, dest_stat.st_uid, dest_stat.st_gid)
            if hasattr(os, 'chflags') and hasattr(dest_stat, 'st_flags'):
                try:
                    os.chflags(b_src, dest_stat.st_flags)
                except OSError as e:
                    for err in ('EOPNOTSUPP', 'ENOTSUP'):
                        if hasattr(errno, err) and e.errno == getattr(errno, err):
                            break
                    else:
                        raise
        except OSError as e:
            if e.errno != errno.EPERM:
                raise
        if self.selinux_enabled():
            context = self.selinux_context(dest)
    elif self.selinux_enabled():
        context = self.selinux_default_context(dest)
    creating = not os.path.exists(b_dest)
    try:
        os.rename(b_src, b_dest)
    except (IOError, OSError) as e:
        if e.errno not in [errno.EPERM, errno.EXDEV, errno.EACCES, errno.ETXTBSY, errno.EBUSY]:
            self.fail_json(msg='Could not replace file: %s to %s: %s' % (src, dest, to_native(e)), exception=traceback.format_exc())
        else:
            b_dest_dir = os.path.dirname(b_dest)
            b_suffix = os.path.basename(b_dest)
            error_msg = None
            tmp_dest_name = None
            try:
                tmp_dest_fd, tmp_dest_name = tempfile.mkstemp(prefix=b'.ansible_tmp', dir=b_dest_dir, suffix=b_suffix)
            except (OSError, IOError) as e:
                error_msg = 'The destination directory (%s) is not writable by the current user. Error was: %s' % (os.path.dirname(dest), to_native(e))
            finally:
                if error_msg:
                    if unsafe_writes:
                        self._unsafe_writes(b_src, b_dest)
                    else:
                        self.fail_json(msg=error_msg, exception=traceback.format_exc())
            if tmp_dest_name:
                b_tmp_dest_name = to_bytes(tmp_dest_name, errors='surrogate_or_strict')
                try:
                    try:
                        os.close(tmp_dest_fd)
                        try:
                            shutil.move(b_src, b_tmp_dest_name)
                        except OSError:
                            shutil.copy2(b_src, b_tmp_dest_name)
                        if self.selinux_enabled():
                            self.set_context_if_different(b_tmp_dest_name, context, False)
                        try:
                            tmp_stat = os.stat(b_tmp_dest_name)
                            if dest_stat and (tmp_stat.st_uid != dest_stat.st_uid or tmp_stat.st_gid != dest_stat.st_gid):
                                os.chown(b_tmp_dest_name, dest_stat.st_uid, dest_stat.st_gid)
                        except OSError as e:
                            if e.errno != errno.EPERM:
                                raise
                        try:
                            os.rename(b_tmp_dest_name, b_dest)
                        except (shutil.Error, OSError, IOError) as e:
                            if unsafe_writes and e.errno == errno.EBUSY:
                                self._unsafe_writes(b_tmp_dest_name, b_dest)
                            else:
                                self.fail_json(msg='Unable to make %s into to %s, failed final rename from %s: %s' % (src, dest, b_tmp_dest_name, to_native(e)), exception=traceback.format_exc())
                    except (shutil.Error, OSError, IOError) as e:
                        if unsafe_writes:
                            self._unsafe_writes(b_src, b_dest)
                        else:
                            self.fail_json(msg='Failed to replace file: %s to %s: %s' % (src, dest, to_native(e)), exception=traceback.format_exc())
                finally:
                    self.cleanup(b_tmp_dest_name)
    if creating:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(b_dest, DEFAULT_PERM & ~umask)
        try:
            os.chown(b_dest, os.geteuid(), os.getegid())
        except OSError:
            pass
    if self.selinux_enabled():
        self.set_context_if_different(dest, context, False)