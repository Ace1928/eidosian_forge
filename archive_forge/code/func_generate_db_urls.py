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
def generate_db_urls(db_urls, extra_drivers):
    """Generate a set of URLs to test given configured URLs plus additional
    driver names.

    Given::

        --dburi postgresql://db1          --dburi postgresql://db2          --dburi postgresql://db2          --dbdriver=psycopg2 --dbdriver=asyncpg?async_fallback=true

    Noting that the default postgresql driver is psycopg2,  the output
    would be::

        postgresql+psycopg2://db1
        postgresql+asyncpg://db1
        postgresql+psycopg2://db2
        postgresql+psycopg2://db3

    That is, for the driver in a --dburi, we want to keep that and use that
    driver for each URL it's part of .   For a driver that is only
    in --dbdrivers, we want to use it just once for one of the URLs.
    for a driver that is both coming from --dburi as well as --dbdrivers,
    we want to keep it in that dburi.

    Driver specific query options can be specified by added them to the
    driver name. For example, to enable the async fallback option for
    asyncpg::

        --dburi postgresql://db1          --dbdriver=asyncpg?async_fallback=true

    """
    urls = set()
    backend_to_driver_we_already_have = collections.defaultdict(set)
    urls_plus_dialects = [(url_obj, url_obj.get_dialect()) for url_obj in [sa_url.make_url(db_url) for db_url in db_urls]]
    for url_obj, dialect in urls_plus_dialects:
        driver_name = url_obj.get_driver_name()
        backend_to_driver_we_already_have[dialect.name].add(driver_name)
    backend_to_driver_we_need = {}
    for url_obj, dialect in urls_plus_dialects:
        backend = dialect.name
        dialect.load_provisioning()
        if backend not in backend_to_driver_we_need:
            backend_to_driver_we_need[backend] = extra_per_backend = set(extra_drivers).difference(backend_to_driver_we_already_have[backend])
        else:
            extra_per_backend = backend_to_driver_we_need[backend]
        for driver_url in _generate_driver_urls(url_obj, extra_per_backend):
            if driver_url in urls:
                continue
            urls.add(driver_url)
            yield driver_url