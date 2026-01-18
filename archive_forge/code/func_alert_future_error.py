from __future__ import annotations
import typing as ty
import warnings
from .deprecator import Deprecator
from .pkg_info import cmp_pkg_version
def alert_future_error(msg: str, version: str, *, warning_class: type[Warning]=FutureWarning, error_class: type[Exception]=RuntimeError, warning_rec: str='', error_rec: str='', stacklevel: int=2) -> None:
    """Warn or error with appropriate messages for changing functionality.

    Parameters
    ----------
    msg : str
        Description of the condition that led to the alert
    version : str
        NiBabel version at which the warning will become an error
    warning_class : subclass of Warning, optional
        Warning class to emit before version
    error_class : subclass of Exception, optional
        Error class to emit after version
    warning_rec : str, optional
        Guidance for suppressing the warning and avoiding the future error
    error_rec: str, optional
        Guidance for resolving the error
    stacklevel: int, optional
        Warnings stacklevel to provide; note that this will be incremented by
        1, so provide the stacklevel you would provide directly to warnings.warn()
    """
    if cmp_pkg_version(version) > 0:
        msg = f'{msg} This will error in NiBabel {version}. {warning_rec}'
        warnings.warn(msg.strip(), warning_class, stacklevel=stacklevel + 1)
    else:
        raise error_class(f'{msg} {error_rec}'.strip())