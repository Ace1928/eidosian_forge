import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log
def process_template_line(self, line):
    action, patterns, dir, dir_pattern = self._parse_template_line(line)
    if action == 'include':
        self.debug_print('include ' + ' '.join(patterns))
        for pattern in patterns:
            if not self.include_pattern(pattern, anchor=1):
                log.warn("warning: no files found matching '%s'", pattern)
    elif action == 'exclude':
        self.debug_print('exclude ' + ' '.join(patterns))
        for pattern in patterns:
            if not self.exclude_pattern(pattern, anchor=1):
                log.warn("warning: no previously-included files found matching '%s'", pattern)
    elif action == 'global-include':
        self.debug_print('global-include ' + ' '.join(patterns))
        for pattern in patterns:
            if not self.include_pattern(pattern, anchor=0):
                log.warn("warning: no files found matching '%s' anywhere in distribution", pattern)
    elif action == 'global-exclude':
        self.debug_print('global-exclude ' + ' '.join(patterns))
        for pattern in patterns:
            if not self.exclude_pattern(pattern, anchor=0):
                log.warn("warning: no previously-included files matching '%s' found anywhere in distribution", pattern)
    elif action == 'recursive-include':
        self.debug_print('recursive-include %s %s' % (dir, ' '.join(patterns)))
        for pattern in patterns:
            if not self.include_pattern(pattern, prefix=dir):
                log.warn("warning: no files found matching '%s' under directory '%s'", pattern, dir)
    elif action == 'recursive-exclude':
        self.debug_print('recursive-exclude %s %s' % (dir, ' '.join(patterns)))
        for pattern in patterns:
            if not self.exclude_pattern(pattern, prefix=dir):
                log.warn("warning: no previously-included files matching '%s' found under directory '%s'", pattern, dir)
    elif action == 'graft':
        self.debug_print('graft ' + dir_pattern)
        if not self.include_pattern(None, prefix=dir_pattern):
            log.warn("warning: no directories found matching '%s'", dir_pattern)
    elif action == 'prune':
        self.debug_print('prune ' + dir_pattern)
        if not self.exclude_pattern(None, prefix=dir_pattern):
            log.warn("no previously-included directories found matching '%s'", dir_pattern)
    else:
        raise DistutilsInternalError("this cannot happen: invalid action '%s'" % action)