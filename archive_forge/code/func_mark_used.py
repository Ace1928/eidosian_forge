import logging
from typing import Set, Callable
def mark_used(self, uri: str, logger: logging.Logger=default_logger):
    """Mark a URI as in use.  URIs in use will not be deleted."""
    if uri in self._used_uris:
        return
    elif uri in self._unused_uris:
        self._used_uris.add(uri)
        self._unused_uris.remove(uri)
    else:
        raise ValueError(f'Got request to mark URI {uri} used, but this URI is not present in the cache.')
    logger.info(f'Marked URI {uri} used.')
    self._check_valid()