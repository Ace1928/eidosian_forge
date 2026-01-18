from collections.abc import Iterable
from io import BytesIO
import os
import re
import shutil
import sys
import tempfile
from unittest import TestCase as _TestCase
from fontTools.config import Config
from fontTools.misc.textTools import tobytes
from fontTools.misc.xmlWriter import XMLWriter
def parseXmlInto(font, parseInto, xmlSnippet):
    parsed_xml = [e for e in parseXML(xmlSnippet.strip()) if not isinstance(e, str)]
    for name, attrs, content in parsed_xml:
        parseInto.fromXML(name, attrs, content, font)
    parseInto.populateDefaults()
    return parseInto