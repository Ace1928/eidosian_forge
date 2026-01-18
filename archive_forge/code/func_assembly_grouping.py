import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def assembly_grouping(self, partition_lengths, delimiter):
    digit_groups = ['%s' * length for length in partition_lengths]
    return delimiter.join(digit_groups)