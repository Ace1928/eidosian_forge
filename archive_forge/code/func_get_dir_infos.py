import inspect
import types
import traceback
import sys
import operator as op
from collections import namedtuple
import warnings
import re
import builtins
import typing
from pathlib import Path
from typing import Optional, Tuple
from jedi.inference.compiled.getattr_static import getattr_static
def get_dir_infos(self):
    """
        Used to return a couple of infos that are needed when accessing the sub
        objects of an objects
        """
    tuples = dict(((name, self.is_allowed_getattr(name)) for name in self.dir()))
    return (self.needs_type_completions(), tuples)