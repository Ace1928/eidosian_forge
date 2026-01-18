from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def get_flag_suggestions(attempt, longopt_list):
    """Returns helpful similar matches for an invalid flag."""
    if len(attempt) <= 2 or not longopt_list:
        return []
    option_names = [v.split('=')[0] for v in longopt_list]
    distances = [(_damerau_levenshtein(attempt, option[0:len(attempt)]), option) for option in option_names]
    distances.sort()
    least_errors, _ = distances[0]
    if least_errors >= _SUGGESTION_ERROR_RATE_THRESHOLD * len(attempt):
        return []
    suggestions = []
    for errors, name in distances:
        if errors == least_errors:
            suggestions.append(name)
        else:
            break
    return suggestions