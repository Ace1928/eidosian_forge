from __future__ import annotations
import itertools
import shutil
import os
import textwrap
import typing as T
import collections
from . import build
from . import coredata
from . import environment
from . import mesonlib
from . import mintro
from . import mlog
from .ast import AstIDGenerator, IntrospectionInterpreter
from .mesonlib import MachineChoice, OptionKey
def run_impl(options: CMDOptions, builddir: str) -> int:
    print_only = not options.cmd_line_options and (not options.clearcache)
    c = None
    try:
        c = Conf(builddir)
        if c.default_values_only and (not print_only):
            raise mesonlib.MesonException('No valid build directory found, cannot modify options.')
        if c.default_values_only or print_only:
            c.print_conf(options.pager)
            return 0
        save = False
        if options.cmd_line_options:
            save = c.set_options(options.cmd_line_options)
            coredata.update_cmd_line_file(builddir, options)
        if options.clearcache:
            c.clear_cache()
            save = True
        if save:
            c.save()
            mintro.update_build_options(c.coredata, c.build.environment.info_dir)
            mintro.write_meson_info_file(c.build, [])
    except ConfException as e:
        mlog.log('Meson configurator encountered an error:')
        if c is not None and c.build is not None:
            mintro.write_meson_info_file(c.build, [e])
        raise e
    except BrokenPipeError:
        pass
    return 0