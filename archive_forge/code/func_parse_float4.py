import re
from kivy.logger import Logger
from kivy.resources import resource_find
def parse_float4(text):
    """Parse a string to a list of exactly 4 floats.

        >>> parse_float4('54 87. 35 0')
        54, 87., 35, 0

    """
    texts = [x for x in text.split(' ') if x.strip() != '']
    value = list(map(parse_float, texts))
    if len(value) < 1:
        raise Exception('Invalid float4 format: %s' % text)
    elif len(value) == 1:
        return [value[0] for x in range(4)]
    elif len(value) == 2:
        return [value[0], value[1], value[0], value[1]]
    elif len(value) == 3:
        return [value[0], value[1], value[0], value[2]]
    elif len(value) > 4:
        raise Exception('Too many values in %s' % text)
    return value