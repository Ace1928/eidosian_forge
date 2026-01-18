import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
def split_opt(opt: str) -> t.Tuple[str, str]:
    first = opt[:1]
    if first.isalnum():
        return ('', opt)
    if opt[1:2] == first:
        return (opt[:2], opt[2:])
    return (first, opt[1:])