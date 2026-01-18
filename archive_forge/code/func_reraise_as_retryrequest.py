import functools
from oslo_db import exception as db_exc
from oslo_utils import excutils
import sqlalchemy
from sqlalchemy.ext import associationproxy
from sqlalchemy.orm import exc
from sqlalchemy.orm import properties
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib import exceptions as n_exc
def reraise_as_retryrequest(function):
    """Wrap the said function with a RetryRequest upon error.

    :param function: The function to wrap/decorate.
    :returns: The 'function' wrapped in a try block that will reraise any
        Exception's as a RetryRequest.
    :raises RetryRequest: If the wrapped function raises retriable exception.
    """

    @functools.wraps(function)
    def _wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            with excutils.save_and_reraise_exception() as ctx:
                if is_retriable(e):
                    ctx.reraise = False
                    raise db_exc.RetryRequest(e)
    return _wrapped