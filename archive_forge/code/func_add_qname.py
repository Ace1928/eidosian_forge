import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def add_qname(qname):
    try:
        if qname[:1] == '{':
            uri, tag = qname[1:].rsplit('}', 1)
            prefix = namespaces.get(uri)
            if prefix is None:
                prefix = _namespace_map.get(uri)
                if prefix is None:
                    prefix = 'ns%d' % len(namespaces)
                if prefix != 'xml':
                    namespaces[uri] = prefix
            if prefix:
                qnames[qname] = '%s:%s' % (prefix, tag)
            else:
                qnames[qname] = tag
        else:
            if default_namespace:
                raise ValueError('cannot use non-qualified names with default_namespace option')
            qnames[qname] = qname
    except TypeError:
        _raise_serialization_error(qname)