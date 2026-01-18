from __future__ import unicode_literals
import argparse
import io
import logging
import os
import sys
import textwrap
import cmakelang
from cmakelang.lint import lintdb
from tangent.tooling.gendoc import format_directive
def gen_table(outfile):
    write_ruler(outfile)
    for idstr, msgfmt, _ in lintdb.LINT_DB:
        write_cell(outfile, idstr, msgfmt)
        write_ruler(outfile)
    outfile.write('\n')