from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
def annotations_validator(annotations: T.List[T.Union[str, T.List[str]]]) -> T.Optional[str]:
    """Validate gdbus-codegen annotations argument"""
    badlist = 'must be made up of 3 strings for ELEMENT, KEY, and VALUE'
    if not annotations:
        return None
    elif all((isinstance(annot, str) for annot in annotations)):
        if len(annotations) == 3:
            return None
        else:
            return badlist
    elif not all((isinstance(annot, list) for annot in annotations)):
        for c, annot in enumerate(annotations):
            if not isinstance(annot, list):
                return f'element {c + 1} must be a list'
    else:
        for c, annot in enumerate(annotations):
            if len(annot) != 3 or not all((isinstance(i, str) for i in annot)):
                return f'element {c + 1} {badlist}'
    return None