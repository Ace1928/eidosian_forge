from __future__ import unicode_literals
import re
from builtins import str
from commonmark.common import escape_xml
from commonmark.render.renderer import Renderer
def potentially_unsafe(url):
    return re.search(reUnsafeProtocol, url) and (not re.search(reSafeDataProtocol, url))