import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
def measure_length(text, term):
    """
    .. deprecated:: 1.12.0.

    :rtype: int
    :returns: Length of the first sequence in the string
    """
    try:
        text, capability = next(iter_parse(term, text))
        if capability:
            return len(text)
    except StopIteration:
        return 0
    return 0