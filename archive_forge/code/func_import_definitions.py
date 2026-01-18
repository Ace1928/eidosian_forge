from suds import *
from suds.bindings.document import Document
from suds.bindings.rpc import RPC, Encoded
from suds.reader import DocumentReader
from suds.sax.element import Element
from suds.sudsobject import Object, Facade, Metadata
from suds.xsd import qualify, Namespace
from suds.xsd.query import ElementQuery
from suds.xsd.schema import Schema, SchemaCollection
import re
from . import soaparray
from urllib.parse import urljoin
from logging import getLogger
def import_definitions(self, definitions, d):
    """Import/merge WSDL definitions."""
    definitions.types += d.types
    definitions.messages.update(d.messages)
    definitions.port_types.update(d.port_types)
    definitions.bindings.update(d.bindings)
    self.imported = d
    log.debug('imported (WSDL):\n%s', d)