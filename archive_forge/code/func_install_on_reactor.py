from typing import Any
from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import (
from zope.interface.declarations import implementer, provider
from scrapy.utils.datatypes import LocalCache
def install_on_reactor(self):
    self.reactor.installNameResolver(self)