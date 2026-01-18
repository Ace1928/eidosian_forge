from __future__ import unicode_literals
import codecs
from email import message_from_file
import json
import logging
import re
from . import DistlibException, __version__
from .compat import StringIO, string_types, text_type
from .markers import interpret
from .util import extract_by_key, get_extras
from .version import get_scheme, PEP440_VERSION_RE
def add_requirements(self, requirements):
    if self._legacy:
        self._legacy.add_requirements(requirements)
    else:
        run_requires = self._data.setdefault('run_requires', [])
        always = None
        for entry in run_requires:
            if 'environment' not in entry and 'extra' not in entry:
                always = entry
                break
        if always is None:
            always = {'requires': requirements}
            run_requires.insert(0, always)
        else:
            rset = set(always['requires']) | set(requirements)
            always['requires'] = sorted(rset)