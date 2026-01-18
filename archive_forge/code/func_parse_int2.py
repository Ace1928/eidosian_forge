import re
from kivy.logger import Logger
from kivy.resources import resource_find
def parse_int2(text):
    """Parse a string to a list of exactly 2 integers.

        >>> print(parse_int2("12 54"))
        12, 54

    """
    texts = [x for x in text.split(' ') if x.strip() != '']
    value = list(map(parse_int, texts))
    if len(value) < 1:
        raise Exception('Invalid int2 format: %s' % text)
    elif len(value) == 1:
        return [value[0], value[0]]
    elif len(value) > 2:
        raise Exception('Too many values in %s: %s' % (text, str(value)))
    return value