from __future__ import (absolute_import, division, print_function)
import abc
import os
from ansible.module_utils import six
def get_fingerprint_from_file(gpg_runner, path):
    if not os.path.exists(path):
        raise GPGError('{path} does not exist'.format(path=path))
    stdout = gpg_runner.run_command(['--no-keyring', '--with-colons', '--import-options', 'show-only', '--import', path], check_rc=True)[1]
    return get_fingerprint_from_stdout(stdout)