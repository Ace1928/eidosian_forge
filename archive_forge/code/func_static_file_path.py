import os
from traitlets import (
from jupyter_core.paths import jupyter_path
from jupyter_server.transutils import _i18n
from jupyter_server.utils import url_path_join
@property
def static_file_path(self):
    """return extra paths + the default location"""
    return self.extra_static_paths