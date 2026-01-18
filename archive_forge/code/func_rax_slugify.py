from __future__ import (absolute_import, division, print_function)
import os
import re
from uuid import UUID
from ansible.module_utils.six import text_type, binary_type
def rax_slugify(value):
    """Prepend a key with rax_ and normalize the key name"""
    return 'rax_%s' % re.sub('[^\\w-]', '_', value).lower().lstrip('_')