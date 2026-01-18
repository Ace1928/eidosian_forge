import sys
import os
import inspect
from . import copydir
from . import command
from paste.util.template import paste_script_template_renderer
def template_dir(self):
    assert self._template_dir is not None, "Template %r didn't set _template_dir" % self
    if isinstance(self._template_dir, tuple):
        return self._template_dir
    else:
        return os.path.join(self.module_dir(), self._template_dir)