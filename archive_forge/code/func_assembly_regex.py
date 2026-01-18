import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def assembly_regex(self, partition_lengths, delimiter, strict):
    regex = '\\D*(\\d)\\D*' * sum(partition_lengths)
    if strict:
        regex = '^' + regex + '$'
    return regex