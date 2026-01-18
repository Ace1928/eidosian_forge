from __future__ import (absolute_import, division, print_function)
from datetime import datetime as dt
from functools import reduce
import logging
from operator import add
import os
import shutil
import sys
import tempfile
import numpy as np
import pkg_resources
from ..symbolic import SymbolicSys
from .. import __version__
def common_cse_symbols():
    idx = 0
    while True:
        yield self.odesys.be.Symbol('m_p_cse[%d]' % idx)
        idx += 1