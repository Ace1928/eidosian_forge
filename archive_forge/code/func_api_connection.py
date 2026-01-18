from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
@contextmanager
def api_connection(self):
    """
        Execute a given code block after connecting to the API.

        When the block has finished, call :func:`exit_json` to report that the module has finished to Ansible.
        """
    self.connect()
    yield
    self.exit_json()