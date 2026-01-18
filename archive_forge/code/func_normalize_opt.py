import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext
from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError
def normalize_opt(opt: str, ctx: t.Optional['Context']) -> str:
    if ctx is None or ctx.token_normalize_func is None:
        return opt
    prefix, opt = split_opt(opt)
    return f'{prefix}{ctx.token_normalize_func(opt)}'