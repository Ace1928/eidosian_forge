from __future__ import unicode_literals
from __future__ import print_function
import re
import six
import textwrap
from pybtex.exceptions import PybtexError
from pybtex.utils import (
from pybtex.richtext import Text
from pybtex.bibtex.utils import split_tex_string, scan_bibtex_string
from pybtex.errors import report_error
from pybtex.py3compat import fix_unicode_literals_in_doctest, python_2_unicode_compatible
from pybtex.plugin import find_plugin
def process_von_last(parts):
    von_last = parts[:-1]
    definitely_not_von = parts[-1:]
    if von_last:
        von, last = rsplit_at(von_last, is_von_name)
        self.prelast_names.extend(von)
        self.last_names.extend(last)
    self.last_names.extend(definitely_not_von)