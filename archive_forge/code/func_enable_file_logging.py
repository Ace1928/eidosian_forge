import logging
from warnings import warn
import os
import sys
from .misc import str2bool
def enable_file_logging(self):
    config = self._config
    LOG_FILENAME = os.path.join(config.get('logging', 'log_directory'), 'pypeline.log')
    hdlr = RFHandler(LOG_FILENAME, maxBytes=int(config.get('logging', 'log_size')), backupCount=int(config.get('logging', 'log_rotate')))
    formatter = logging.Formatter(fmt=self.fmt, datefmt=self.datefmt)
    hdlr.setFormatter(formatter)
    self._logger.addHandler(hdlr)
    self._utlogger.addHandler(hdlr)
    self._iflogger.addHandler(hdlr)
    self._fmlogger.addHandler(hdlr)
    self._hdlr = hdlr