from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
from pasta.base import annotate
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def optional_token(self, node, attr_name, token_val, allow_whitespace_prefix=False, default=False):
    del allow_whitespace_prefix
    value = fmt.get(node, attr_name)
    if value is None and default:
        value = token_val
    self.code += value or ''