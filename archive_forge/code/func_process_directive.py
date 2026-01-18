import fnmatch
import logging
import os
import re
import sys
from . import DistlibException
from .compat import fsdecode
from .util import convert_path
def process_directive(self, directive):
    """
        Process a directive which either adds some files from ``allfiles`` to
        ``files``, or removes some files from ``files``.

        :param directive: The directive to process. This should be in a format
                     compatible with distutils ``MANIFEST.in`` files:

                     http://docs.python.org/distutils/sourcedist.html#commands
        """
    action, patterns, thedir, dirpattern = self._parse_directive(directive)
    if action == 'include':
        for pattern in patterns:
            if not self._include_pattern(pattern, anchor=True):
                logger.warning('no files found matching %r', pattern)
    elif action == 'exclude':
        for pattern in patterns:
            self._exclude_pattern(pattern, anchor=True)
    elif action == 'global-include':
        for pattern in patterns:
            if not self._include_pattern(pattern, anchor=False):
                logger.warning('no files found matching %r anywhere in distribution', pattern)
    elif action == 'global-exclude':
        for pattern in patterns:
            self._exclude_pattern(pattern, anchor=False)
    elif action == 'recursive-include':
        for pattern in patterns:
            if not self._include_pattern(pattern, prefix=thedir):
                logger.warning('no files found matching %r under directory %r', pattern, thedir)
    elif action == 'recursive-exclude':
        for pattern in patterns:
            self._exclude_pattern(pattern, prefix=thedir)
    elif action == 'graft':
        if not self._include_pattern(None, prefix=dirpattern):
            logger.warning('no directories found matching %r', dirpattern)
    elif action == 'prune':
        if not self._exclude_pattern(None, prefix=dirpattern):
            logger.warning('no previously-included directories found matching %r', dirpattern)
    else:
        raise DistlibException('invalid action %r' % action)