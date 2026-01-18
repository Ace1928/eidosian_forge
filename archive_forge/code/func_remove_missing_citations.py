from __future__ import print_function, unicode_literals
from collections import defaultdict
import six
from pybtex.bibtex.builtins import builtins, print_warning
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.bibtex.utils import wrap
from pybtex.utils import CaseInsensitiveDict
def remove_missing_citations(self, citations):
    for citation in citations:
        if citation in self.bib_data.entries:
            yield citation
        else:
            print_warning('missing database entry for "{0}"'.format(citation))