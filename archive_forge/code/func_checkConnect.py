import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def checkConnect(self):
    """Check the connect/disconnect synchronous calls."""
    conn = self.dbpool.connect()
    self.checkOpenfunCalled(conn)
    curs = conn.cursor()
    curs.execute('insert into simple(x) values(1)')
    curs.execute('select x from simple')
    res = curs.fetchall()
    self.assertEqual(len(res), 1)
    self.assertEqual(len(res[0]), 1)
    self.assertEqual(res[0][0], 1)
    curs.execute('delete from simple')
    curs.execute('select x from simple')
    self.assertEqual(len(curs.fetchall()), 0)
    curs.close()
    self.dbpool.disconnect(conn)