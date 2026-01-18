import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def makeSQLTests(base, suffix, globals):
    """
    Make a test case for every db connector which can connect.

    @param base: Base class for test case. Additional base classes
                 will be a DBConnector subclass and unittest.TestCase
    @param suffix: A suffix used to create test case names. Prefixes
                   are defined in the DBConnector subclasses.
    """
    connectors = [PySQLite2Connector, SQLite3Connector, PyPgSQLConnector, PsycopgConnector, MySQLConnector, FirebirdConnector]
    tests = {}
    for connclass in connectors:
        name = connclass.TEST_PREFIX + suffix

        class testcase(connclass, base, unittest.TestCase):
            __module__ = connclass.__module__
        testcase.__name__ = name
        if hasattr(connclass, '__qualname__'):
            testcase.__qualname__ = '.'.join(connclass.__qualname__.split()[0:-1] + [name])
        tests[name] = testcase
    globals.update(tests)