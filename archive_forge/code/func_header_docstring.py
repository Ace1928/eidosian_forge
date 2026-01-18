import warnings
from webob.compat import (
from webob.headers import _trans_key
def header_docstring(header, rfc_section):
    if header.isupper():
        header = _trans_key(header)
    major_section = rfc_section.split('.')[0]
    link = 'http://www.w3.org/Protocols/rfc2616/rfc2616-sec%s.html#sec%s' % (major_section, rfc_section)
    return 'Gets and sets the ``%s`` header (`HTTP spec section %s <%s>`_).' % (header, rfc_section, link)