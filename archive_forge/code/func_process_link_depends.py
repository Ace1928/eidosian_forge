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
def process_link_depends(self, sources):
    """Process the link_depends keyword argument.

        This is designed to handle strings, Files, and the output of Custom
        Targets. Notably it doesn't handle generator() returned objects, since
        adding them as a link depends would inherently cause them to be
        generated twice, since the output needs to be passed to the ld_args and
        link_depends.
        """
    sources = listify(sources)
    for s in sources:
        if isinstance(s, File):
            self.link_depends.append(s)
        elif isinstance(s, str):
            self.link_depends.append(File.from_source_file(self.environment.source_dir, self.subdir, s))
        elif hasattr(s, 'get_outputs'):
            self.link_depends.append(s)
        else:
            raise InvalidArguments('Link_depends arguments must be strings, Files, or a Custom Target, or lists thereof.')