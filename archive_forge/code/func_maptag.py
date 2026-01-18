from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
@classmethod
def maptag(cls, tag, fn):
    """
        Map (override) tag => I{class} mapping.

        @param tag: An XSD tag name.
        @type tag: str
        @param fn: A function or class.
        @type fn: fn|class.

        """
    cls.tags[tag] = fn