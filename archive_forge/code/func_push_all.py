import os
from ... import osutils
from . import wrapper
def push_all(self, quiet=None, force=None, refresh=None):
    return wrapper.quilt_push_all(self.tree.basedir, patches_dir=self.patches_dir, series_file=self.series_file, quiet=quiet, force=force, refresh=refresh)