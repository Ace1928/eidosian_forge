from __future__ import annotations
import argparse, datetime, glob, json, os, platform, shutil, sys, tempfile, time
import cProfile as profile
from pathlib import Path
import typing as T
from . import build, coredata, environment, interpreter, mesonlib, mintro, mlog
from .mesonlib import MesonException
def run_genvslite_setup(options: CMDOptions) -> None:
    builddir_prefix = options.builddir
    genvsliteval = options.cmd_line_options.pop(mesonlib.OptionKey('genvslite'))
    k_backend = mesonlib.OptionKey('backend')
    if k_backend in options.cmd_line_options.keys():
        if options.cmd_line_options[k_backend] != 'ninja':
            raise MesonException("Explicitly specifying a backend option with 'genvslite' is not necessary (the ninja backend is always used) but specifying a non-ninja backend conflicts with a 'genvslite' setup")
    else:
        options.cmd_line_options[k_backend] = 'ninja'
    buildtypes_list = coredata.get_genvs_default_buildtype_list()
    vslite_ctx = {}
    for buildtypestr in buildtypes_list:
        options.builddir = f'{builddir_prefix}_{buildtypestr}'
        options.cmd_line_options[mesonlib.OptionKey('buildtype')] = buildtypestr
        app = MesonApp(options)
        vslite_ctx[buildtypestr] = app.generate(capture=True)
    options.builddir = f'{builddir_prefix}_vs'
    options.cmd_line_options[mesonlib.OptionKey('genvslite')] = genvsliteval
    app = MesonApp(options)
    app.generate(capture=False, vslite_ctx=vslite_ctx)