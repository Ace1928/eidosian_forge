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
def set_wrapped(self):
    """Set (wrapped|bare) flag on messages."""
    for b in list(self.bindings.values()):
        for op in list(b.operations.values()):
            for body in (op.soap.input.body, op.soap.output.body):
                body.wrapped = False
                if not self.options.unwrap:
                    continue
                if len(body.parts) != 1:
                    continue
                for p in body.parts:
                    if p.element is None:
                        continue
                    query = ElementQuery(p.element)
                    pt = query.execute(self.schema)
                    if pt is None:
                        raise TypeNotFound(query.ref)
                    resolved = pt.resolve()
                    if resolved.builtin():
                        continue
                    body.wrapped = True