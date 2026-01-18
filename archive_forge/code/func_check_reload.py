import os
import sys
import time
import threading
import traceback
from paste.util.classinstance import classinstancemethod
def check_reload(self):
    filenames = list(self.extra_files)
    for file_callback in self.file_callbacks:
        try:
            filenames.extend(file_callback())
        except:
            print('Error calling paste.reloader callback %r:' % file_callback, file=sys.stderr)
            traceback.print_exc()
    for module in sys.modules.values():
        try:
            filename = module.__file__
        except (AttributeError, ImportError):
            continue
        if filename is not None:
            filenames.append(filename)
    for filename in filenames:
        try:
            stat = os.stat(filename)
            if stat:
                mtime = stat.st_mtime
            else:
                mtime = 0
        except (OSError, IOError):
            continue
        if filename.endswith('.pyc') and os.path.exists(filename[:-1]):
            mtime = max(os.stat(filename[:-1]).st_mtime, mtime)
        elif filename.endswith('$py.class') and os.path.exists(filename[:-9] + '.py'):
            mtime = max(os.stat(filename[:-9] + '.py').st_mtime, mtime)
        if filename not in self.module_mtimes:
            self.module_mtimes[filename] = mtime
        elif self.module_mtimes[filename] < mtime:
            print('%s changed; reloading...' % filename, file=sys.stderr)
            return False
    return True