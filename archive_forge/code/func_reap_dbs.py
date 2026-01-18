from __future__ import annotations
import collections
import logging
from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema
def reap_dbs(idents_file):
    log.info('Reaping databases...')
    urls = collections.defaultdict(set)
    idents = collections.defaultdict(set)
    dialects = {}
    with open(idents_file) as file_:
        for line in file_:
            line = line.strip()
            db_name, db_url = line.split(' ')
            url_obj = sa_url.make_url(db_url)
            if db_name not in dialects:
                dialects[db_name] = url_obj.get_dialect()
                dialects[db_name].load_provisioning()
            url_key = (url_obj.get_backend_name(), url_obj.host)
            urls[url_key].add(db_url)
            idents[url_key].add(db_name)
    for url_key in urls:
        url = list(urls[url_key])[0]
        ident = idents[url_key]
        run_reap_dbs(url, ident)