from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_native
def check_for_import():
    if IMPORT_ERROR:
        raise AnsibleFilterError('isodate python package is required:  %s' % IMPORT_ERROR)