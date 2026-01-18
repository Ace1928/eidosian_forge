from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
def finalize_main_c_code(self):
    self.close_global_decls()
    code = self.parts['utility_code_def']
    util = TempitaUtilityCode.load_cached('TypeConversions', 'TypeConversion.c')
    code.put(util.format_code(util.impl))
    code.putln('')
    code = self.parts['utility_code_pragmas']
    util = UtilityCode.load_cached('UtilityCodePragmas', 'ModuleSetupCode.c')
    code.putln(util.format_code(util.impl))
    code.putln('')
    code = self.parts['utility_code_pragmas_end']
    util = UtilityCode.load_cached('UtilityCodePragmasEnd', 'ModuleSetupCode.c')
    code.putln(util.format_code(util.impl))
    code.putln('')