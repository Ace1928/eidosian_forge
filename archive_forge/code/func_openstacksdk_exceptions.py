import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
@contextlib.contextmanager
def openstacksdk_exceptions(error_message=None):
    """Context manager for dealing with openstack exceptions.

    :param string error_message: String to use for the exception message
        content on non-SDKException exception.

        Useful for avoiding wrapping SDKException exceptions
        within themselves. Code called from within the context may throw such
        exceptions without having to catch and reraise them.

        Non-SDKException exceptions thrown within the context will
        be wrapped and the exception message will be appended to the given
        error message.
    """
    try:
        yield
    except exceptions.SDKException:
        raise
    except Exception as e:
        if error_message is None:
            error_message = str(e)
        raise exceptions.SDKException(error_message)