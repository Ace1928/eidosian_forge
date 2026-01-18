from __future__ import annotations
import sys
from .compression import decompress
from .exceptions import MessageStateError, reraise
from .serialization import loads
from .utils.functional import dictfilter
def reject_log_error(self, logger, errors, requeue=False):
    try:
        self.reject(requeue=requeue)
    except errors as exc:
        logger.critical("Couldn't reject %r, reason: %r", self.delivery_tag, exc, exc_info=True)