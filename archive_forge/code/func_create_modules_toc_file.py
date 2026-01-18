import argparse
import glob
import locale
import os
import sys
from copy import copy
from fnmatch import fnmatch
from importlib.machinery import EXTENSION_SUFFIXES
from os import path
from typing import Any, Generator, List, Optional, Tuple
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.cmd.quickstart import EXTENSIONS
from sphinx.locale import __
from sphinx.util.osutil import FileAvoidWrite, ensuredir
from sphinx.util.template import ReSTRenderer
def create_modules_toc_file(modules: List[str], opts: Any, name: str='modules', user_template_dir: Optional[str]=None) -> None:
    """Create the module's index."""
    modules.sort()
    prev_module = ''
    for module in modules[:]:
        if module.startswith(prev_module + '.'):
            modules.remove(module)
        else:
            prev_module = module
    context = {'header': opts.header, 'maxdepth': opts.maxdepth, 'docnames': modules}
    text = ReSTRenderer([user_template_dir, template_dir]).render('toc.rst_t', context)
    write_file(name, text, opts)