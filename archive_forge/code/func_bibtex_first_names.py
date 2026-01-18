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
@property
@fix_unicode_literals_in_doctest
def bibtex_first_names(self):
    """A list of first and middle names together.
        (BibTeX treats all middle names as first.)

        .. versionadded:: 0.19
            Earlier versions used :py:meth:`Person.bibtex_first`, which is now deprecated.


        >>> knuth = Person('Donald E. Knuth')
        >>> knuth.bibtex_first_names
        [u'Donald', u'E.']
        """
    return self.first_names + self.middle_names