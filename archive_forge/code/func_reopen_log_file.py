import datetime
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import ovs.dirs
import ovs.unixctl
import ovs.util
@staticmethod
def reopen_log_file():
    """Closes and then attempts to re-open the current log file.  (This is
        useful just after log rotation, to ensure that the new log file starts
        being used.)"""
    if Vlog.__log_file:
        logger = logging.getLogger('file')
        logger.removeHandler(Vlog.__file_handler)
        Vlog.__file_handler = logging.FileHandler(Vlog.__log_file)
        logger.addHandler(Vlog.__file_handler)