import logging
from typing import Set, Callable
def mark_unused(self, uri: str, logger: logging.Logger=default_logger):
    """Mark a URI as unused and okay to be deleted."""
    if uri not in self._used_uris:
        logger.info(f'URI {uri} is already unused.')
    else:
        self._unused_uris.add(uri)
        self._used_uris.remove(uri)
    logger.info(f'Marked URI {uri} unused.')
    self._evict_if_needed(logger)
    self._check_valid()