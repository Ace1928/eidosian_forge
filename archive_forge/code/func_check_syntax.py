from pythran.tables import MODULES
from pythran.intrinsic import Class
from pythran.typing import Tuple, List, Set, Dict
from pythran.utils import isstr
from pythran import metadata
import beniget
import gast as ast
import logging
import numpy as np
def check_syntax(node):
    """Does nothing but raising PythranSyntaxError when needed"""
    SyntaxChecker().visit(node)