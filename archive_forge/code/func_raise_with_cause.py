import functools
import io
import logging
import os
import sys
import time
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from oslo_utils import timeutils
def raise_with_cause(exc_cls, message, *args, **kwargs):
    """Helper to raise + chain exceptions (when able) and associate a *cause*.

    NOTE(harlowja): Since in py3.x exceptions can be chained (due to
    :pep:`3134`) we should try to raise the desired exception with the given
    *cause* (or extract a *cause* from the current stack if able) so that the
    exception formats nicely in old and new versions of python. Since py2.x
    does **not** support exception chaining (or formatting) the exception
    class provided should take a ``cause`` keyword argument (which it may
    discard if it wants) to its constructor which can then be
    inspected/retained on py2.x to get *similar* information as would be
    automatically included/obtainable in py3.x.

    :param exc_cls: the exception class to raise (typically one derived
                    from :py:class:`.CausedByException` or equivalent).
    :param message: the text/str message that will be passed to
                    the exceptions constructor as its first positional
                    argument.
    :param args: any additional positional arguments to pass to the
                 exceptions constructor.
    :param kwargs: any additional keyword arguments to pass to the
                   exceptions constructor.

    .. versionadded:: 1.6
    """
    if 'cause' not in kwargs:
        exc_type, exc, exc_tb = sys.exc_info()
        try:
            if exc is not None:
                kwargs['cause'] = exc
        finally:
            del (exc_type, exc, exc_tb)
    raise exc_cls(message, *args, **kwargs) from kwargs.get('cause')