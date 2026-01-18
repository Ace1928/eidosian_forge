from __future__ import (absolute_import, division, print_function)
import random
import re
import string
import sys
from collections import namedtuple
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.utils.display import Display
def random_salt(length=8):
    """Return a text string suitable for use as a salt for the hash functions we use to encrypt passwords.
    """
    salt_chars = string.ascii_letters + string.digits + u'./'
    return random_password(length=length, chars=salt_chars)