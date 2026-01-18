from __future__ import absolute_import, print_function, division
import os
import tempfile
import pytest
from petl.test.helpers import ieq
import petl as etl
from petl.io.whoosh import fromtextindex, totextindex, appendtextindex, \
def test_fromindex_index():
    dirname = tempfile.mkdtemp()
    schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
    ix = create_in(dirname, schema)
    writer = ix.writer()
    writer.add_document(title=u'First document', path=u'/a', content=u"This is the first document we've added!")
    writer.add_document(title=u'Second document', path=u'/b', content=u'The second one is even more interesting!')
    writer.commit()
    expect = ((u'path', u'title'), (u'/a', u'First document'), (u'/b', u'Second document'))
    actual = fromtextindex(ix)
    ieq(expect, actual)