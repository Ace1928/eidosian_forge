from io import BytesIO
from io import StringIO
from lxml import etree
from bs4.element import (
from bs4.builder import (
from bs4.dammit import EncodingDetector
def default_parser(self, encoding):
    return etree.HTMLParser