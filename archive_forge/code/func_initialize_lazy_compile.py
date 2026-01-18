from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.util import lazy_regex_patterns
def initialize_lazy_compile():
    re.compile = _lazy_compile