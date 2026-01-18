from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def fill_form_html(html, values, form_id=None, form_index=None):
    result_type = type(html)
    if isinstance(html, basestring):
        doc = fromstring(html)
    else:
        doc = copy.deepcopy(html)
    fill_form(doc, values, form_id=form_id, form_index=form_index)
    return _transform_result(result_type, doc)