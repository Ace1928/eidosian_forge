import io
import time
from urllib.request import urlopen
from urllib.parse import quote
from typing import Dict, List
from Bio._utils import function_with_previous
def search_iter(db, query, limit=None, batch=100):
    """Call TogoWS search iterating over the results (generator function).

    Arguments:
     - db - database (string), see http://togows.dbcls.jp/search
     - query - search term (string)
     - limit - optional upper bound on number of search results
     - batch - number of search results to pull back each time talk to
       TogoWS (currently limited to 100).

    You would use this function within a for loop, e.g.

    >>> from Bio import TogoWS
    >>> for id in TogoWS.search_iter("pubmed", "diabetes+human", limit=10):
    ...     print("PubMed ID: %s" %id) # maybe fetch data with entry?
    PubMed ID: ...

    Internally this first calls the Bio.TogoWS.search_count() and then
    uses Bio.TogoWS.search() to get the results in batches.
    """
    count = search_count(db, query)
    if not count:
        return
    remain = count
    if limit is not None:
        remain = min(remain, limit)
    offset = 1
    prev_ids = []
    while remain:
        batch = min(batch, remain)
        ids = search(db, query, offset, batch).read().strip().split()
        assert len(ids) == batch, 'Got %i, expected %i' % (len(ids), batch)
        if ids == prev_ids:
            raise RuntimeError('Same search results for previous offset')
        for identifier in ids:
            if identifier in prev_ids:
                raise RuntimeError(f'Result {identifier} was in previous batch')
            yield identifier
        offset += batch
        remain -= batch
        prev_ids = ids