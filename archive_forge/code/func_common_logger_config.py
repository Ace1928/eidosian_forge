import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def common_logger_config(self, logger, config, incremental=False):
    """
        Perform configuration which is common to root and non-root loggers.
        """
    level = config.get('level', None)
    if level is not None:
        logger.setLevel(logging._checkLevel(level))
    if not incremental:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        handlers = config.get('handlers', None)
        if handlers:
            self.add_handlers(logger, handlers)
        filters = config.get('filters', None)
        if filters:
            self.add_filters(logger, filters)