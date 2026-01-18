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
def help_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--help`` option which immediately prints the help page
    and exits the program.

    This is usually unnecessary, as the ``--help`` option is added to
    each command automatically unless ``add_help_option=False`` is
    passed.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--help"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """

    def callback(ctx: Context, param: Parameter, value: bool) -> None:
        if not value or ctx.resilient_parsing:
            return
        echo(ctx.get_help(), color=ctx.color)
        ctx.exit()
    if not param_decls:
        param_decls = ('--help',)
    kwargs.setdefault('is_flag', True)
    kwargs.setdefault('expose_value', False)
    kwargs.setdefault('is_eager', True)
    kwargs.setdefault('help', _('Show this message and exit.'))
    kwargs['callback'] = callback
    return option(*param_decls, **kwargs)