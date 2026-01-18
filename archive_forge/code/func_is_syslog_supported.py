import logging
import logging.handlers
import os
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import on_macos, on_windows
from coloredlogs import (
def is_syslog_supported():
    """
    Determine whether system logging is supported.

    :returns:

        :data:`True` if system logging is supported and can be enabled,
        :data:`False` if system logging is not supported or there are good
        reasons for not enabling it.

    The decision making process here is as follows:

    Override
     If the environment variable ``$COLOREDLOGS_SYSLOG`` is set it is evaluated
     using :func:`~humanfriendly.coerce_boolean()` and the resulting value
     overrides the platform detection discussed below, this allows users to
     override the decision making process if they disagree / know better.

    Linux / UNIX
     On systems that are not Windows or MacOS (see below) we assume UNIX which
     means either syslog is available or sending a bunch of UDP packets to
     nowhere won't hurt anyone...

    Microsoft Windows
     Over the years I've had multiple reports of :pypi:`coloredlogs` spewing
     extremely verbose errno 10057 warning messages to the console (once for
     each log message I suppose) so I now assume it a default that
     "syslog-style system logging" is not generally available on Windows.

    Apple MacOS
     There's cPython issue `#38780`_ which seems to result in a fatal exception
     when the Python interpreter shuts down. This is (way) worse than not
     having system logging enabled. The error message mentioned in `#38780`_
     has actually been following me around for years now, see for example:

     - https://github.com/xolox/python-rotate-backups/issues/9 mentions Docker
       images implying Linux, so not strictly the same as `#38780`_.

     - https://github.com/xolox/python-npm-accel/issues/4 is definitely related
       to `#38780`_ and is what eventually prompted me to add the
       :func:`is_syslog_supported()` logic.

    .. _#38780: https://bugs.python.org/issue38780
    """
    override = os.environ.get('COLOREDLOGS_SYSLOG')
    if override is not None:
        return coerce_boolean(override)
    else:
        return not (on_windows() or on_macos())