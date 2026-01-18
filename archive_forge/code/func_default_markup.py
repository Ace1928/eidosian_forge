import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def default_markup(text, version):
    return '<span title="%s">%s</span>' % (html_escape(_unicode(version), 1), text)