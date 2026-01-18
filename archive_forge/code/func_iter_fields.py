from __future__ import absolute_import
import binascii
import codecs
import os
from io import BytesIO
from .fields import RequestField
from .packages import six
from .packages.six import b
def iter_fields(fields):
    """
    .. deprecated:: 1.6

    Iterate over fields.

    The addition of :class:`~urllib3.fields.RequestField` makes this function
    obsolete. Instead, use :func:`iter_field_objects`, which returns
    :class:`~urllib3.fields.RequestField` objects.

    Supports list of (k, v) tuples and dicts.
    """
    if isinstance(fields, dict):
        return ((k, v) for k, v in six.iteritems(fields))
    return ((k, v) for k, v in fields)