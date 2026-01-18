import logging
import sys
import warnings
def set_warning_filter(log_level):
    if log_level == logging.ERROR:
        warnings.simplefilter('ignore')
    elif log_level == logging.WARNING:
        warnings.simplefilter('ignore')
    elif log_level == logging.INFO:
        warnings.simplefilter('once')