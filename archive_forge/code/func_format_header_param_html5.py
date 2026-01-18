from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def format_header_param_html5(name, value):
    """
    Helper function to format and quote a single header parameter using the
    HTML5 strategy.

    Particularly useful for header parameters which might contain
    non-ASCII values, like file names. This follows the `HTML5 Working Draft
    Section 4.10.22.7`_ and matches the behavior of curl and modern browsers.

    .. _HTML5 Working Draft Section 4.10.22.7:
        https://w3c.github.io/html/sec-forms.html#multipart-form-data

    :param name:
        The name of the parameter, a string expected to be ASCII only.
    :param value:
        The value of the parameter, provided as ``bytes`` or `str``.
    :ret:
        A unicode string, stripped of troublesome characters.
    """
    if isinstance(value, six.binary_type):
        value = value.decode('utf-8')
    value = _replace_multiple(value, _HTML5_REPLACEMENTS)
    return u'%s="%s"' % (name, value)