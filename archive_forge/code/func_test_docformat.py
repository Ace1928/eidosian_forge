import sys
import pytest
from numpy.testing import assert_equal, suppress_warnings
from scipy._lib import doccer
def test_docformat():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        udd = doccer.unindent_dict(doc_dict)
        formatted = doccer.docformat(docstring, udd)
        assert_equal(formatted, filled_docstring)
        single_doc = 'Single line doc %(strtest1)s'
        formatted = doccer.docformat(single_doc, doc_dict)
        assert_equal(formatted, 'Single line doc Another test\n   with some indent')