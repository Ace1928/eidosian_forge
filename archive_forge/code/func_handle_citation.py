from __future__ import unicode_literals, with_statement
import re
import pybtex.io
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
from pybtex import py3compat
def handle_citation(self, keys):
    for key in keys.split(','):
        key_lower = key.lower()
        if key_lower in self._canonical_keys:
            existing_key = self._canonical_keys[key_lower]
            if key != existing_key:
                msg = 'case mismatch error between cite keys {0} and {1}'
                report_error(AuxDataError(msg.format(key, existing_key), self.context))
        self.citations.append(key)
        self._canonical_keys[key_lower] = key