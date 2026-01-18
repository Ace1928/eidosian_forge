from __future__ import annotations
import select
import sys
from typing import Callable
from twisted.internet import default
from twisted.internet.default import _getInstallFunction, install
from twisted.internet.interfaces import IReactorCore
from twisted.internet.test.test_main import NoReactor
from twisted.python.reflect import requireModule
from twisted.python.runtime import Platform
from twisted.trial.unittest import SynchronousTestCase
def installer() -> object:
    installed.append(True)
    return install()