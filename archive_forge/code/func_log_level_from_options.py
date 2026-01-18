import logging
import sys
import warnings
def log_level_from_options(options):
    log_level = logging.WARNING
    if options.verbose_level == 0:
        log_level = logging.ERROR
    elif options.verbose_level == 2:
        log_level = logging.INFO
    elif options.verbose_level >= 3:
        log_level = logging.DEBUG
    return log_level