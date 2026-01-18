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
def create_package_file(root: str, master_package: str, subroot: str, py_files: List[str], opts: Any, subs: List[str], is_namespace: bool, excludes: List[str]=[], user_template_dir: Optional[str]=None) -> None:
    """Build the text of the file and write the file."""
    subpackages = [module_join(master_package, subroot, pkgname) for pkgname in subs if not is_skipped_package(path.join(root, pkgname), opts, excludes)]
    submodules = [sub.split('.')[0] for sub in py_files if not is_skipped_module(path.join(root, sub), opts, excludes) and (not is_initpy(sub))]
    submodules = sorted(set(submodules))
    submodules = [module_join(master_package, subroot, modname) for modname in submodules]
    options = copy(OPTIONS)
    if opts.includeprivate and 'private-members' not in options:
        options.append('private-members')
    pkgname = module_join(master_package, subroot)
    context = {'pkgname': pkgname, 'subpackages': subpackages, 'submodules': submodules, 'is_namespace': is_namespace, 'modulefirst': opts.modulefirst, 'separatemodules': opts.separatemodules, 'automodule_options': options, 'show_headings': not opts.noheadings, 'maxdepth': opts.maxdepth}
    text = ReSTRenderer([user_template_dir, template_dir]).render('package.rst_t', context)
    write_file(pkgname, text, opts)
    if submodules and opts.separatemodules:
        for submodule in submodules:
            create_module_file(None, submodule, opts, user_template_dir)