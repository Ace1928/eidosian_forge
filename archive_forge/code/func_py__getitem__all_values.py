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
def py__getitem__all_values(self):
    if isinstance(self._obj, dict):
        return [self._create_access_path(v) for v in self._obj.values()]
    if isinstance(self._obj, (list, tuple)):
        return [self._create_access_path(v) for v in self._obj]
    if self.is_instance():
        cls = DirectObjectAccess(self._inference_state, self._obj.__class__)
        return cls.py__getitem__all_values()
    try:
        getitem = self._obj.__getitem__
    except AttributeError:
        pass
    else:
        annotation = DirectObjectAccess(self._inference_state, getitem).get_return_annotation()
        if annotation is not None:
            return [annotation]
    return None