from __future__ import absolute_import, division, unicode_literals
from xml.dom import minidom, Node
import weakref
from . import base
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def documentClass(self):
    self.dom = Dom.getDOMImplementation().createDocument(None, None, None)
    return weakref.proxy(self)