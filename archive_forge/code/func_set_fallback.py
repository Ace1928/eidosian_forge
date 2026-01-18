from __future__ import annotations
from .interpreterobjects import extract_required_kwarg
from .. import mlog
from .. import dependencies
from .. import build
from ..wrap import WrapMode
from ..mesonlib import OptionKey, extract_as_list, stringlistify, version_compare_many, listify
from ..dependencies import Dependency, DependencyException, NotFoundDependency
from ..interpreterbase import (MesonInterpreterObject, FeatureNew,
import typing as T
def set_fallback(self, fbinfo: T.Optional[T.Union[T.List[str], str]]) -> None:
    if fbinfo is None:
        return
    if self.allow_fallback is not None:
        raise InvalidArguments('"fallback" and "allow_fallback" arguments are mutually exclusive')
    fbinfo = stringlistify(fbinfo)
    if len(fbinfo) == 0:
        self.allow_fallback = False
        return
    if len(fbinfo) == 1:
        FeatureNew.single_use('Fallback without variable name', '0.53.0', self.subproject)
        subp_name, varname = (fbinfo[0], None)
    elif len(fbinfo) == 2:
        subp_name, varname = fbinfo
    else:
        raise InterpreterException('Fallback info must have one or two items.')
    self._subproject_impl(subp_name, varname)