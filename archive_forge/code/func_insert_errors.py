from lxml.etree import XPath, ElementBase
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import _forms_xpath, _options_xpath, _nons, _transform_result
from lxml.html import defs
import copy
def insert_errors(el, errors, form_id=None, form_index=None, error_class='error', error_creator=default_error_creator):
    el = _find_form(el, form_id=form_id, form_index=form_index)
    for name, error in errors.items():
        if error is None:
            continue
        for error_el, message in _find_elements_for_name(el, name, error):
            assert isinstance(message, (basestring, type(None), ElementBase)), 'Bad message: %r' % message
            _insert_error(error_el, message, error_class, error_creator)