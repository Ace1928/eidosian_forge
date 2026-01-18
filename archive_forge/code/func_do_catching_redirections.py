import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def do_catching_redirections(action: Callable[[Transport], T], transport: Transport, redirected: Callable[[Transport, errors.RedirectRequested, str], Transport]) -> T:
    """Execute an action with given transport catching redirections.

    This is a facility provided for callers needing to follow redirections
    silently. The silence is relative: it is the caller responsability to
    inform the user about each redirection or only inform the user of a user
    via the exception parameter.

    Args:
      action: A callable, what the caller want to do while catching
                  redirections.
      transport: The initial transport used.
      redirected: A callable receiving the redirected transport and the
                  RedirectRequested exception.

    :return: Whatever 'action' returns
    """
    MAX_REDIRECTIONS = 8
    for redirections in range(MAX_REDIRECTIONS):
        try:
            return action(transport)
        except errors.RedirectRequested as e:
            redirection_notice = '{} is{} redirected to {}'.format(e.source, e.permanently, e.target)
            transport = redirected(transport, e, redirection_notice)
    else:
        raise errors.TooManyRedirections