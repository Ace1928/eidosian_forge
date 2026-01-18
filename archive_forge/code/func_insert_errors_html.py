from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def insert_errors_html(html, values, **kw):
    result_type = type(html)
    if isinstance(html, basestring):
        doc = fromstring(html)
    else:
        doc = copy.deepcopy(html)
    insert_errors(doc, values, **kw)
    return _transform_result(result_type, doc)