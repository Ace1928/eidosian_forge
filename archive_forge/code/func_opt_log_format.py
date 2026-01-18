import typing
from sys import stderr, stdout
from textwrap import dedent
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, cast
from twisted.copyright import version
from twisted.internet.interfaces import IReactorCore
from twisted.logger import (
from twisted.plugin import getPlugins
from twisted.python.usage import Options, UsageError
from ..reactors import NoSuchReactor, getReactorTypes, installReactor
from ..runner._exit import ExitStatus, exit
from ..service import IServiceMaker
def opt_log_format(self, format: str) -> None:
    """
        Log file format.
        (options: "text", "json"; default: "text" if the log file is a tty,
        otherwise "json")
        """
    format = format.lower()
    if format == 'text':
        self['fileLogObserverFactory'] = textFileLogObserver
    elif format == 'json':
        self['fileLogObserverFactory'] = jsonFileLogObserver
    else:
        raise UsageError(f'Invalid log format: {format}')
    self['logFormat'] = format