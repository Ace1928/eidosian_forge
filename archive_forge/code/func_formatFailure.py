import io
import linecache
import warnings
from collections import OrderedDict
from html import escape
from typing import (
from xml.sax import handler, make_parser
from xml.sax.xmlreader import AttributesNSImpl, Locator
from zope.interface import implementer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import urlpath
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import fullyQualifiedName
from twisted.web import resource
from twisted.web._element import Element, renderer
from twisted.web._flatten import Flattenable, flatten, flattenString
from twisted.web._stan import CDATA, Comment, Tag, slot
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
def formatFailure(myFailure):
    """
    Construct an HTML representation of the given failure.

    Consider using L{FailureElement} instead.

    @type myFailure: L{Failure<twisted.python.failure.Failure>}

    @rtype: L{bytes}
    @return: A string containing the HTML representation of the given failure.
    """
    result = []
    flattenString(None, FailureElement(myFailure)).addBoth(result.append)
    if isinstance(result[0], bytes):
        return result[0].decode('utf-8').encode('ascii', 'xmlcharrefreplace')
    result[0].raiseException()