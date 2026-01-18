from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
def itertextindex(index_or_dirname, indexname, docnum_field):
    import whoosh.index
    if isinstance(index_or_dirname, string_types):
        dirname = index_or_dirname
        index = whoosh.index.open_dir(dirname, indexname=indexname, readonly=True)
        needs_closing = True
    elif isinstance(index_or_dirname, whoosh.index.Index):
        index = index_or_dirname
        needs_closing = False
    else:
        raise ArgumentError('expected string or index, found %r' % index_or_dirname)
    try:
        if docnum_field is None:
            hdr = tuple(index.schema.stored_names())
            yield hdr
            astuple = operator.itemgetter(*index.schema.stored_names())
            for _, stored_fields_dict in index.reader().iter_docs():
                yield astuple(stored_fields_dict)
        else:
            hdr = (docnum_field,) + tuple(index.schema.stored_names())
            yield hdr
            astuple = operator.itemgetter(*index.schema.stored_names())
            for docnum, stored_fields_dict in index.reader().iter_docs():
                yield ((docnum,) + astuple(stored_fields_dict))
    except:
        raise
    finally:
        if needs_closing:
            index.close()