from __future__ import (absolute_import, division, print_function)
import abc
import os
from ansible.module_utils import six
def get_fingerprint_from_stdout(stdout):
    lines = stdout.splitlines(False)
    for line in lines:
        if line.startswith('fpr:'):
            parts = line.split(':')
            if len(parts) <= 9 or not parts[9]:
                raise GPGError('Result line "{line}" does not have fingerprint as 10th component'.format(line=line))
            return parts[9]
    raise GPGError('Cannot extract fingerprint from stdout "{stdout}"'.format(stdout=stdout))