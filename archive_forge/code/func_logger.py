from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
@property
def logger(self):
    try:
        return self._logger
    except AttributeError:
        self._logger = _get_logger(self.name)
        self.level = self._logger.level
        return self._logger