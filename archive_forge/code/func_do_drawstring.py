import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def do_drawstring(self, drawstring, drawobj, texlbl_name='texlbl', use_drawstring_pos=False):
    """Parse and draw drawsting

        Just a wrapper around do_draw_op.
        """
    drawoperations, stat = parse_drawstring(drawstring)
    return self.do_draw_op(drawoperations, drawobj, stat, texlbl_name, use_drawstring_pos)