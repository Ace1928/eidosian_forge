from __future__ import print_function, unicode_literals, with_statement
import re
import sys
from os import path
from shutil import rmtree
from subprocess import PIPE, Popen
from tempfile import mkdtemp
from pybtex.database import BibliographyData, Entry, Person
from pybtex.database.output import bibtex
from pybtex.errors import report_error
from pybtex.exceptions import PybtexError
def run_bibtex(style, database, citations=None):
    if citations is None:
        citations = list(database.entries.keys())
    tmpdir = mkdtemp(prefix='pybtex_test_')
    try:
        write_bib(path.join(tmpdir, 'test.bib'), database)
        write_aux(path.join(tmpdir, 'test.aux'), citations)
        write_bst(path.join(tmpdir, 'test.bst'), style)
        bibtex = Popen(('bibtex', 'test'), cwd=tmpdir, stdout=PIPE, stderr=PIPE)
        stdout, stderr = bibtex.communicate()
        if bibtex.returncode:
            report_error(PybtexError(stdout))
        with open(path.join(tmpdir, 'test.bbl')) as bbl_file:
            result = bbl_file.read()
        return result
    finally:
        pass
        rmtree(tmpdir)