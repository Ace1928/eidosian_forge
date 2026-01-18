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
def opt_reactor(self, name: str) -> None:
    """
        The name of the reactor to use.
        (options: {options})
        """
    try:
        self['reactor'] = self.installReactor(name)
    except NoSuchReactor:
        raise UsageError(f'Unknown reactor: {name}')
    else:
        self['reactorName'] = name