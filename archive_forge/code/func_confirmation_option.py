import inspect
import types
import typing as t
from functools import update_wrapper
from gettext import gettext as _
from .core import Argument
from .core import Command
from .core import Context
from .core import Group
from .core import Option
from .core import Parameter
from .globals import get_current_context
from .utils import echo
def confirmation_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--yes`` option which shows a prompt before continuing if
    not passed. If the prompt is declined, the program will exit.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--yes"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """

    def callback(ctx: Context, param: Parameter, value: bool) -> None:
        if not value:
            ctx.abort()
    if not param_decls:
        param_decls = ('--yes',)
    kwargs.setdefault('is_flag', True)
    kwargs.setdefault('callback', callback)
    kwargs.setdefault('expose_value', False)
    kwargs.setdefault('prompt', 'Do you want to continue?')
    kwargs.setdefault('help', 'Confirm the action without prompting.')
    return option(*param_decls, **kwargs)