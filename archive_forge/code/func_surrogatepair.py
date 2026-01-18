import re
from io import TextIOWrapper
def surrogatepair(c):
    """Given a unicode character code with length greater than 16 bits,
    return the two 16 bit surrogate pair.
    """
    return (55232 + (c >> 10), 56320 + (c & 1023))