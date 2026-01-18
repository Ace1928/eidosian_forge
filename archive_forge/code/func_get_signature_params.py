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
def get_signature_params(self):
    return [SignatureParam(name=p.name, has_default=p.default is not p.empty, default=self._create_access_path(p.default), default_string=repr(p.default), has_annotation=p.annotation is not p.empty, annotation=self._create_access_path(p.annotation), annotation_string=self._annotation_to_str(p.annotation), kind_name=str(p.kind)) for p in self._get_signature().parameters.values()]