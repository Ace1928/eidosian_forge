from __future__ import annotations
import json
from logging import Logger
import requests
import tornado
from jupyter_server.base.handlers import APIHandler
def fetch_listings(logger: Logger | None) -> None:
    """Fetch the listings for the extension manager."""
    if not logger:
        from traitlets import log
        logger = log.get_logger()
    assert logger is not None
    if len(ListingsHandler.blocked_extensions_uris) > 0:
        blocked_extensions = []
        for blocked_extensions_uri in ListingsHandler.blocked_extensions_uris:
            logger.info('Fetching blocked_extensions from %s', ListingsHandler.blocked_extensions_uris)
            r = requests.request('GET', blocked_extensions_uri, **ListingsHandler.listings_request_opts)
            j = json.loads(r.text)
            for b in j['blocked_extensions']:
                blocked_extensions.append(b)
            ListingsHandler.blocked_extensions = blocked_extensions
    if len(ListingsHandler.allowed_extensions_uris) > 0:
        allowed_extensions = []
        for allowed_extensions_uri in ListingsHandler.allowed_extensions_uris:
            logger.info('Fetching allowed_extensions from %s', ListingsHandler.allowed_extensions_uris)
            r = requests.request('GET', allowed_extensions_uri, **ListingsHandler.listings_request_opts)
            j = json.loads(r.text)
            for w in j['allowed_extensions']:
                allowed_extensions.append(w)
        ListingsHandler.allowed_extensions = allowed_extensions
    ListingsHandler.listings = json.dumps({'blocked_extensions_uris': list(ListingsHandler.blocked_extensions_uris), 'allowed_extensions_uris': list(ListingsHandler.allowed_extensions_uris), 'blocked_extensions': ListingsHandler.blocked_extensions, 'allowed_extensions': ListingsHandler.allowed_extensions})