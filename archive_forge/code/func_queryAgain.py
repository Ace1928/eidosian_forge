import socket
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.logger import Logger
from twisted.names import dns
from twisted.names.error import (
def queryAgain(records):
    ans, auth, add = records
    return extractRecord(nsResolver, name, ans + auth + add, level - 1)