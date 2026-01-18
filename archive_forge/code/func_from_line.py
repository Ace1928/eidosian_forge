from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
@classmethod
def from_line(cls, line: bytes) -> 'RefLogEntry':
    """:return: New RefLogEntry instance from the given revlog line.

        :param line: Line bytes without trailing newline

        :raise ValueError: If `line` could not be parsed
        """
    line_str = line.decode(defenc)
    fields = line_str.split('\t', 1)
    if len(fields) == 1:
        info, msg = (fields[0], None)
    elif len(fields) == 2:
        info, msg = fields
    else:
        raise ValueError('Line must have up to two TAB-separated fields. Got %s' % repr(line_str))
    oldhexsha = info[:40]
    newhexsha = info[41:81]
    for hexsha in (oldhexsha, newhexsha):
        if not cls._re_hexsha_only.match(hexsha):
            raise ValueError('Invalid hexsha: %r' % (hexsha,))
    email_end = info.find('>', 82)
    if email_end == -1:
        raise ValueError('Missing token: >')
    actor = Actor._from_string(info[82:email_end + 1])
    time, tz_offset = parse_date(info[email_end + 2:])
    return RefLogEntry((oldhexsha, newhexsha, actor, (time, tz_offset), msg))