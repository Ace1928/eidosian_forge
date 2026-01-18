from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noArgsFlattening
@noKwargs
@typed_pos_args('str.format', varargs=object)
def format_method(self, args: T.Tuple[T.List[TYPE_var]], kwargs: TYPE_kwargs) -> str:
    arg_strings: T.List[str] = []
    for arg in args[0]:
        try:
            arg_strings.append(stringifyUserArguments(arg, self.subproject))
        except InvalidArguments as e:
            FeatureBroken.single_use(f'str.format: {str(e)}', '1.3.0', self.subproject, location=self.current_node)
            arg_strings.append(str(arg))

    def arg_replace(match: T.Match[str]) -> str:
        idx = int(match.group(1))
        if idx >= len(arg_strings):
            raise InvalidArguments(f'Format placeholder @{idx}@ out of range.')
        return arg_strings[idx]
    return re.sub('@(\\d+)@', arg_replace, self.held_object)