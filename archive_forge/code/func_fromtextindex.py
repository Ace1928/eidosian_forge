from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
def fromtextindex(index_or_dirname, indexname=None, docnum_field=None):
    """
    Extract all documents from a Whoosh index. E.g.::

        >>> import petl as etl
        >>> import os
        >>> # set up an index and load some documents via the Whoosh API
        ... from whoosh.index import create_in
        >>> from whoosh.fields import *
        >>> schema = Schema(title=TEXT(stored=True), path=ID(stored=True),
        ...                 content=TEXT)
        >>> dirname = 'example.whoosh'
        >>> if not os.path.exists(dirname):
        ...     os.mkdir(dirname)
        ...
        >>> index = create_in(dirname, schema)
        >>> writer = index.writer()
        >>> writer.add_document(title=u"First document", path=u"/a",
        ...                     content=u"This is the first document we've added!")
        >>> writer.add_document(title=u"Second document", path=u"/b",
        ...                     content=u"The second one is even more interesting!")
        >>> writer.commit()
        >>> # extract documents as a table
        ... table = etl.fromtextindex(dirname)
        >>> table
        +------+-------------------+
        | path | title             |
        +======+===================+
        | '/a' | 'First document'  |
        +------+-------------------+
        | '/b' | 'Second document' |
        +------+-------------------+

    Keyword arguments:
    
    index_or_dirname
        Either an instance of `whoosh.index.Index` or a string containing the
        directory path where the index is stored.
    indexname
        String containing the name of the index, if multiple indexes are stored
        in the same directory.
    docnum_field
        If not None, an extra field will be added to the output table containing
        the internal document number stored in the index. The name of the field
        will be the value of this argument.

    """
    return TextIndexView(index_or_dirname, indexname=indexname, docnum_field=docnum_field)