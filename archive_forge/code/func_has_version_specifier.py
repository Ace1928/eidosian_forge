from __future__ import absolute_import, division, print_function
import argparse
import os
import re
import sys
import tempfile
import operator
import shlex
import traceback
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule, is_executable, missing_required_lib
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.six import PY3
@property
def has_version_specifier(self):
    if self._plain_package:
        return bool(getattr(self._requirement, 'specifier', None) or getattr(self._requirement, 'specs', None))
    return False