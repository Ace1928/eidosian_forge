import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def parse_range(textrange):
    """Parse a patch range, handling the "1" special-case

    :param textrange: The text to parse
    :type textrange: str
    :return: the position and range, as a tuple
    :rtype: (int, int)
    """
    tmp = textrange.split(b',')
    if len(tmp) == 1:
        pos = tmp[0]
        brange = b'1'
    else:
        pos, brange = tmp
    pos = int(pos)
    range = int(brange)
    return (pos, range)