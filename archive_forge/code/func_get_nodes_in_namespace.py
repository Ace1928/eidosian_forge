import datetime
import decimal
import re
from typing import (
from xml.dom.minidom import Document, parseString
from xml.dom.minidom import Element as XmlElement
from xml.parsers.expat import ExpatError
from ._utils import StreamType, deprecate_no_replacement
from .errors import PdfReadError
from .generic import ContentStream, PdfObject
def get_nodes_in_namespace(self, about_uri: str, namespace: str) -> Iterator[Any]:
    for desc in self.rdf_root.getElementsByTagNameNS(RDF_NAMESPACE, 'Description'):
        if desc.getAttributeNS(RDF_NAMESPACE, 'about') == about_uri:
            for i in range(desc.attributes.length):
                attr = desc.attributes.item(i)
                if attr.namespaceURI == namespace:
                    yield attr
            for child in desc.childNodes:
                if child.namespaceURI == namespace:
                    yield child