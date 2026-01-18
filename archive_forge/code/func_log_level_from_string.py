import logging
import sys
import warnings
def log_level_from_string(level_string):
    log_level = {'critical': logging.CRITICAL, 'error': logging.ERROR, 'warning': logging.WARNING, 'info': logging.INFO, 'debug': logging.DEBUG}.get(level_string, logging.WARNING)
    return log_level