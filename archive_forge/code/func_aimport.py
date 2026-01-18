imported with ``from foo import ...`` was also updated.
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
import os
import sys
import traceback
import types
import weakref
import gc
import logging
from importlib import import_module, reload
from importlib.util import source_from_cache
@line_magic
def aimport(self, parameter_s='', stream=None):
    """%aimport => Import modules for automatic reloading.

        %aimport
        List modules to automatically import and not to import.

        %aimport foo
        Import module 'foo' and mark it to be autoreloaded for %autoreload explicit

        %aimport foo, bar
        Import modules 'foo', 'bar' and mark them to be autoreloaded for %autoreload explicit

        %aimport -foo, bar
        Mark module 'foo' to not be autoreloaded for %autoreload explicit, all, or complete, and 'bar'
        to be autoreloaded for mode explicit.
        """
    modname = parameter_s
    if not modname:
        to_reload = sorted(self._reloader.modules.keys())
        to_skip = sorted(self._reloader.skip_modules.keys())
        if stream is None:
            stream = sys.stdout
        if self._reloader.check_all:
            stream.write('Modules to reload:\nall-except-skipped\n')
        else:
            stream.write('Modules to reload:\n%s\n' % ' '.join(to_reload))
        stream.write('\nModules to skip:\n%s\n' % ' '.join(to_skip))
    else:
        for _module in [_.strip() for _ in modname.split(',')]:
            if _module.startswith('-'):
                _module = _module[1:].strip()
                self._reloader.mark_module_skipped(_module)
            else:
                top_module, top_name = self._reloader.aimport_module(_module)
                self.shell.push({top_name: top_module})