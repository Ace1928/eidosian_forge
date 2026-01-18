import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
def get_as_string(self, key):
    keyl = key.lower()
    if keyl in self._multivalued_fields:
        fd = io.StringIO()
        if hasattr(self[key], 'keys'):
            array = [self[key]]
        else:
            fd.write('\n')
            array = self[key]
        order = self._multivalued_fields[keyl]
        field_lengths = {}
        try:
            field_lengths = self._fixed_field_lengths
        except AttributeError:
            pass
        for item in array:
            for x in order:
                raw_value = str(item[x])
                try:
                    length = field_lengths[keyl][x]
                except KeyError:
                    value = raw_value
                else:
                    value = (length - len(raw_value)) * ' ' + raw_value
                if '\n' in value:
                    raise ValueError("'\\n' not allowed in component of multivalued field %s" % key)
                fd.write(' %s' % value)
            fd.write('\n')
        return fd.getvalue().rstrip('\n')
    return Deb822.get_as_string(self, key)