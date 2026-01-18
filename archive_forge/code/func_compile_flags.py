import argparse
import logging
import os
import sys
import pythran
import pythran.types.tog
from distutils.errors import CompileError
def compile_flags(args):
    """
    Build a dictionnary with an entry for cppflags, ldflags, and cxxflags.

    These options are filled according to the command line defined options

    """
    compiler_options = {'define_macros': args.defines, 'undef_macros': args.undefs, 'include_dirs': args.include_dirs, 'extra_compile_args': args.extra_flags, 'library_dirs': args.libraries_dir, 'extra_link_args': args.extra_flags, 'config': args.config}
    for param in ('opts',):
        val = getattr(args, param, None)
        if val:
            compiler_options[param] = val
    return compiler_options