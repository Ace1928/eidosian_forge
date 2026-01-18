import sys
from distutils.log import *  # noqa: F403
from distutils.log import Log as old_Log
from distutils.log import _global_log
from numpy.distutils.misc_util import (red_text, default_text, cyan_text,
def set_threshold(level, force=False):
    prev_level = _global_log.threshold
    if prev_level > DEBUG or force:
        _global_log.threshold = level
        if level <= DEBUG:
            info('set_threshold: setting threshold to DEBUG level, it can be changed only with force argument')
    else:
        info('set_threshold: not changing threshold from DEBUG level %s to %s' % (prev_level, level))
    return prev_level