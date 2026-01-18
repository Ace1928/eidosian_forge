from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
def itersearchindex(index_or_dirname, query, limit, pagenum, pagelen, indexname, docnum_field, score_field, fieldboosts, search_kwargs):
    import whoosh.index
    import whoosh.query
    import whoosh.qparser
    if not search_kwargs:
        search_kwargs = dict()
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
        hdr = tuple()
        if docnum_field is not None:
            hdr += (docnum_field,)
        if score_field is not None:
            hdr += (score_field,)
        stored_names = tuple(index.schema.stored_names())
        hdr += stored_names
        yield hdr
        if isinstance(query, string_types):
            parser = whoosh.qparser.MultifieldParser(index.schema.names(), index.schema, fieldboosts=fieldboosts)
            query = parser.parse(query)
        elif isinstance(query, whoosh.query.Query):
            pass
        else:
            raise ArgumentError('expected string or whoosh.query.Query, found %r' % query)
        astuple = operator.itemgetter(*index.schema.stored_names())
        with index.searcher() as searcher:
            if limit is not None:
                results = searcher.search(query, limit=limit, **search_kwargs)
            else:
                results = searcher.search_page(query, pagenum, pagelen=pagelen, **search_kwargs)
            if docnum_field is None and score_field is None:
                for doc in results:
                    yield astuple(doc)
            else:
                for (docnum, score), doc in izip(results.items(), results):
                    row = tuple()
                    if docnum_field is not None:
                        row += (docnum,)
                    if score_field is not None:
                        row += (score,)
                    row += astuple(doc)
                    yield row
    except:
        raise
    finally:
        if needs_closing:
            index.close()