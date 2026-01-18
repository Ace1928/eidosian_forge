from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
def process_objectlist(self, objects):
    assert isinstance(objects, list)
    deprecated_non_objects = []
    for s in objects:
        if isinstance(s, (str, File, ExtractedObjects)):
            self.objects.append(s)
            if not isinstance(s, ExtractedObjects) and (not is_object(s)):
                deprecated_non_objects.append(s)
        elif isinstance(s, (CustomTarget, CustomTargetIndex, GeneratedList)):
            non_objects = [o for o in s.get_outputs() if not is_object(o)]
            if non_objects:
                raise InvalidArguments(f"Generated file {non_objects[0]} in the 'objects' kwarg is not an object.")
            self.generated.append(s)
        else:
            raise InvalidArguments(f'Bad object of type {type(s).__name__!r} in target {self.name!r}.')
    if deprecated_non_objects:
        FeatureDeprecated.single_use(f"Source file {deprecated_non_objects[0]} in the 'objects' kwarg is not an object.", '1.3.0', self.subproject)