from io import BytesIO
import fastbencode as bencode
from .. import lazy_import
from breezy.bzr import (
from .. import cache_utf8, errors
from .. import revision as _mod_revision
from . import serializer
def read_revision_from_string(self, text):
    ret = bencode.bdecode(text)
    if not isinstance(ret, list):
        raise ValueError('invalid revision text')
    schema = self._schema
    bits = {'timezone': None}
    for key, value in ret:
        var_name, expected_type, validator = schema[key]
        if value.__class__ is not expected_type:
            raise ValueError('key %s did not conform to the expected type %s, but was %s' % (key, expected_type, type(value)))
        if validator is not None:
            value = validator(value)
        bits[var_name] = value
    if len(bits) != len(schema):
        missing = [key for key, (var_name, _, _) in schema.items() if var_name not in bits]
        raise ValueError('Revision text was missing expected keys %s. text %r' % (missing, text))
    del bits[None]
    rev = _mod_revision.Revision(**bits)
    return rev