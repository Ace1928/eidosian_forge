import inspect
import logging
from typing import Optional, Union
from ray.util import log_once
from ray.util.annotations import _mark_annotated
def deprecation_warning(old: str, new: Optional[str]=None, *, help: Optional[str]=None, error: Optional[Union[bool, Exception]]=None) -> None:
    """Warns (via the `logger` object) or throws a deprecation warning/error.

    Args:
        old: A description of the "thing" that is to be deprecated.
        new: A description of the new "thing" that replaces it.
        help: An optional help text to tell the user, what to
            do instead of using `old`.
        error: Whether or which exception to raise. If True, raise ValueError.
            If False, just warn. If `error` is-a subclass of Exception,
            raise that Exception.

    Raises:
        ValueError: If `error=True`.
        Exception: Of type `error`, iff `error` is a sub-class of `Exception`.
    """
    msg = '`{}` has been deprecated.{}'.format(old, ' Use `{}` instead.'.format(new) if new else f' {help}' if help else '')
    if error:
        if not type(error) is bool and issubclass(error, Exception):
            raise error(msg)
        else:
            raise ValueError(msg)
    else:
        logger.warning('DeprecationWarning: ' + msg + ' This will raise an error in the future!')