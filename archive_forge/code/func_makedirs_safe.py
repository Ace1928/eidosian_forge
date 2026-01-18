from __future__ import (absolute_import, division, print_function)
import os
import shutil
from errno import EEXIST
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def makedirs_safe(path, mode=None):
    """
    A *potentially insecure* way to ensure the existence of a directory chain. The "safe" in this function's name
    refers only to its ability to ignore `EEXIST` in the case of multiple callers operating on the same part of
    the directory chain. This function is not safe to use under world-writable locations when the first level of the
    path to be created contains a predictable component. Always create a randomly-named element first if there is any
    chance the parent directory might be world-writable (eg, /tmp) to prevent symlink hijacking and potential
    disclosure or modification of sensitive file contents.

    :arg path: A byte or text string representing a directory chain to be created
    :kwarg mode: If given, the mode to set the directory to
    :raises AnsibleError: If the directory cannot be created and does not already exist.
    :raises UnicodeDecodeError: if the path is not decodable in the utf-8 encoding.
    """
    rpath = unfrackpath(path)
    b_rpath = to_bytes(rpath)
    if not os.path.exists(b_rpath):
        try:
            if mode:
                os.makedirs(b_rpath, mode)
            else:
                os.makedirs(b_rpath)
        except OSError as e:
            if e.errno != EEXIST:
                raise AnsibleError('Unable to create local directories(%s): %s' % (to_native(rpath), to_native(e)))