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
def password_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--password`` option which prompts for a password, hiding
    input and asking to enter the value again for confirmation.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--password"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """
    if not param_decls:
        param_decls = ('--password',)
    kwargs.setdefault('prompt', True)
    kwargs.setdefault('confirmation_prompt', True)
    kwargs.setdefault('hide_input', True)
    return option(*param_decls, **kwargs)