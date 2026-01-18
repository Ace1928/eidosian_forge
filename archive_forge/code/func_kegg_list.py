import io
from urllib.request import urlopen
import time
from Bio._utils import function_with_previous
def kegg_list(database, org=None):
    """KEGG list - Entry list for database, or specified database entries.

    db - database or organism (string)
    org - optional organism (string), see below.

    For the pathway and module databases the optional organism can be
    used to restrict the results.

    """
    if database in ('pathway', 'module') and org:
        resp = _q('list', database, org)
    elif isinstance(database, str) and database and org:
        raise ValueError('Invalid database arg for kegg list request.')
    else:
        if isinstance(database, list):
            if len(database) > 100:
                raise ValueError('Maximum number of databases is 100 for kegg list query')
            database = '+'.join(database)
        resp = _q('list', database)
    return resp