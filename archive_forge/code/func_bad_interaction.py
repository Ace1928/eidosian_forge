import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def bad_interaction(self, transaction):
    if self.can_rollback:
        transaction.execute('insert into simple(x) values(0)')
    transaction.execute('select * from NOTABLE')