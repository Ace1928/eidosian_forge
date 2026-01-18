from __future__ import absolute_import, division, print_function
import base64
import random
import string
from ansible.errors import AnsibleLookupError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_bytes, to_text
@staticmethod
def get_random(random_generator, chars, length):
    if not chars:
        raise AnsibleLookupError('Available characters cannot be None, please change constraints')
    return ''.join((random_generator.choice(chars) for dummy in range(length)))