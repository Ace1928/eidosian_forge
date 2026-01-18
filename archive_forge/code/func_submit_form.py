import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def submit_form(form, extra_values=None, open_http=None):
    """
    Helper function to submit a form.  Returns a file-like object, as from
    ``urllib.urlopen()``.  This object also has a ``.geturl()`` function,
    which shows the URL if there were any redirects.

    You can use this like::

        form = doc.forms[0]
        form.inputs['foo'].value = 'bar' # etc
        response = form.submit()
        doc = parse(response)
        doc.make_links_absolute(response.geturl())

    To change the HTTP requester, pass a function as ``open_http`` keyword
    argument that opens the URL for you.  The function must have the following
    signature::

        open_http(method, URL, values)

    The action is one of 'GET' or 'POST', the URL is the target URL as a
    string, and the values are a sequence of ``(name, value)`` tuples with the
    form data.
    """
    values = form.form_values()
    if extra_values:
        if hasattr(extra_values, 'items'):
            extra_values = extra_values.items()
        values.extend(extra_values)
    if open_http is None:
        open_http = open_http_urllib
    if form.action:
        url = form.action
    else:
        url = form.base_url
    return open_http(form.method, url, values)