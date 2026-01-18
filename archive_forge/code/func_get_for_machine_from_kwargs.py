from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
def get_for_machine_from_kwargs(self, kwargs: T.Dict[str, T.Any]) -> MachineChoice:
    return MachineChoice.BUILD if kwargs.get('native', False) else MachineChoice.HOST