import codecs
import os
import pipes
import re
import subprocess
import tempfile
from humanfriendly.terminal import (
from coloredlogs.converter.colors import (
def parse_hex_color(value):
    """
    Convert a CSS color in hexadecimal notation into its R, G, B components.

    :param value: A CSS color in hexadecimal notation (a string like '#000000').
    :return: A tuple with three integers (with values between 0 and 255)
             corresponding to the R, G and B components of the color.
    :raises: :exc:`~exceptions.ValueError` on values that can't be parsed.
    """
    if value.startswith('#'):
        value = value[1:]
    if len(value) == 3:
        return (int(value[0] * 2, 16), int(value[1] * 2, 16), int(value[2] * 2, 16))
    elif len(value) == 6:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    else:
        raise ValueError()