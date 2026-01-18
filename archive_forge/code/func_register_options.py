import functools
import inspect
import logging
from oslo_config import cfg
from oslo_log._i18n import _
def register_options():
    """Register configuration options used by this library.

    .. note: This is optional since the options are also registered
        automatically when the functions in this module are used.

    """
    CONF.register_opts(deprecated_opts)