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
def add_operations(self, root, definitions):
    """Add <operation/> children."""
    dsop = Element('operation', ns=soapns)
    for c in root.getChildren('operation'):
        op = Facade('Operation')
        op.name = c.get('name')
        sop = c.getChild('operation', default=dsop)
        soap = Facade('soap')
        soap.action = '"%s"' % (sop.get('soapAction', default=''),)
        soap.style = sop.get('style', default=self.soap.style)
        soap.input = Facade('Input')
        soap.input.body = Facade('Body')
        soap.input.headers = []
        soap.output = Facade('Output')
        soap.output.body = Facade('Body')
        soap.output.headers = []
        op.soap = soap
        input = c.getChild('input')
        if input is None:
            input = Element('input', ns=wsdlns)
        body = input.getChild('body')
        self.body(definitions, soap.input.body, body)
        for header in input.getChildren('header'):
            self.header(definitions, soap.input, header)
        output = c.getChild('output')
        if output is None:
            output = Element('output', ns=wsdlns)
        body = output.getChild('body')
        self.body(definitions, soap.output.body, body)
        for header in output.getChildren('header'):
            self.header(definitions, soap.output, header)
        faults = []
        for fault in c.getChildren('fault'):
            sf = fault.getChild('fault')
            if sf is None:
                continue
            fn = fault.get('name')
            f = Facade('Fault')
            f.name = sf.get('name', default=fn)
            f.use = sf.get('use', default='literal')
            faults.append(f)
        soap.faults = faults
        self.operations[op.name] = op