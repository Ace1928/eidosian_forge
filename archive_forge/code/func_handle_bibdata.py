from __future__ import unicode_literals, with_statement
import re
import pybtex.io
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
from pybtex import py3compat
def handle_bibdata(self, bibdata):
    if self.data is not None:
        report_error(AuxDataError('illegal, another \\bibdata command', self.context))
    else:
        self.data = bibdata.split(',')