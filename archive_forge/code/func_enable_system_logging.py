import logging
import logging.handlers
import os
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import on_macos, on_windows
from coloredlogs import (
def enable_system_logging(programname=None, fmt=None, logger=None, reconfigure=True, **kw):
    """
    Redirect :mod:`logging` messages to the system log (e.g. ``/var/log/syslog``).

    :param programname: The program name to embed in log messages (a string, defaults
                         to the result of :func:`~coloredlogs.find_program_name()`).
    :param fmt: The log format for system log messages (a string, defaults to
                :data:`DEFAULT_LOG_FORMAT`).
    :param logger: The logger to which the :class:`~logging.handlers.SysLogHandler`
                   should be connected (defaults to the root logger).
    :param level: The logging level for the :class:`~logging.handlers.SysLogHandler`
                  (defaults to :data:`.DEFAULT_LOG_LEVEL`). This value is coerced
                  using :func:`~coloredlogs.level_to_number()`.
    :param reconfigure: If :data:`True` (the default) multiple calls to
                        :func:`enable_system_logging()` will each override
                        the previous configuration.
    :param kw: Refer to :func:`connect_to_syslog()`.
    :returns: A :class:`~logging.handlers.SysLogHandler` object or
              :data:`None`. If an existing handler is found and `reconfigure`
              is :data:`False` the existing handler object is returned. If the
              connection to the system logging daemon fails :data:`None` is
              returned.

    As of release 15.0 this function uses :func:`is_syslog_supported()` to
    check whether system logging is supported and appropriate before it's
    enabled.

    .. note:: When the logger's effective level is too restrictive it is
              relaxed (refer to `notes about log levels`_ for details).
    """
    if not is_syslog_supported():
        return None
    programname = programname or find_program_name()
    logger = logger or logging.getLogger()
    fmt = fmt or DEFAULT_LOG_FORMAT
    level = level_to_number(kw.get('level', DEFAULT_LOG_LEVEL))
    handler, logger = replace_handler(logger, match_syslog_handler, reconfigure)
    if not (handler and (not reconfigure)):
        handler = connect_to_syslog(**kw)
        if handler:
            ProgramNameFilter.install(handler=handler, fmt=fmt, programname=programname)
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
            adjust_level(logger, level)
    return handler