import logging
import sys
import warnings
def log_level_from_config(config):
    verbose_level = config.get('verbose_level')
    if config.get('debug', False):
        verbose_level = 3
    if verbose_level == 0:
        verbose_level = 'error'
    elif verbose_level == 1:
        verbose_level = config.get('log_level', 'warning')
    elif verbose_level == 2:
        verbose_level = 'info'
    else:
        verbose_level = 'debug'
    return log_level_from_string(verbose_level)