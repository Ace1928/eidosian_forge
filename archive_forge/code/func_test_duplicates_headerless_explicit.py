from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.dedup import duplicates, unique, conflicts, distinct, \
def test_duplicates_headerless_explicit():
    table = []
    with pytest.raises(FieldSelectionError):
        for i in duplicates(table, 'foo'):
            pass