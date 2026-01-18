from __future__ import (absolute_import, division, print_function)
import glob
import os
import pickle
import platform
import select
import shlex
import subprocess
import traceback
from ansible.module_utils.six import PY2, b
from ansible.module_utils.common.text.converters import to_bytes, to_text
def fail_if_missing(module, found, service, msg=''):
    """
    This function will return an error or exit gracefully depending on check mode status
    and if the service is missing or not.

    :arg module: is an  AnsibleModule object, used for it's utility methods
    :arg found: boolean indicating if services was found or not
    :arg service: name of service
    :kw msg: extra info to append to error/success msg when missing
    """
    if not found:
        module.fail_json(msg='Could not find the requested service %s: %s' % (service, msg))