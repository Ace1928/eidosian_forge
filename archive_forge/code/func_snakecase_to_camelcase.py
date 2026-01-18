import itertools
import os
import re
def snakecase_to_camelcase(name):
    """Convert snake-case string to camel-case string."""
    name = _single_underscore_re.split(name)
    name = [_multiple_underscores_re.split(n) for n in name]
    return ''.join((n.capitalize() for n in itertools.chain.from_iterable(name) if n != ''))