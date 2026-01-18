from __future__ import annotations
import itertools
import math
from collections.abc import Mapping, Iterable
from jinja2.filters import pass_environment
from ansible.errors import AnsibleFilterError, AnsibleFilterTypeError
from ansible.module_utils.common.text import formatters
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
def inversepower(x, base=2):
    try:
        if base == 2:
            return math.sqrt(x)
        else:
            return math.pow(x, 1.0 / float(base))
    except (ValueError, TypeError) as e:
        raise AnsibleFilterTypeError('root() can only be used on numbers: %s' % to_native(e))