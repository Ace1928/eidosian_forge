from __future__ import annotations
from .. import environment, mparser, mesonlib
from .baseobjects import (
from .exceptions import (
from .decorators import FeatureNew
from .disabler import Disabler, is_disabled
from .helpers import default_resolve_key, flatten, resolve_second_level_holders, stringifyUserArguments
from .operator import MesonOperator
from ._unholder import _unholder
import os, copy, re, pathlib
import typing as T
import textwrap
def load_root_meson_file(self) -> None:
    mesonfile = os.path.join(self.source_root, self.subdir, environment.build_filename)
    if not os.path.isfile(mesonfile):
        raise InvalidArguments(f'Missing Meson file in {mesonfile}')
    with open(mesonfile, encoding='utf-8') as mf:
        code = mf.read()
    if code.isspace():
        raise InvalidCode('Builder file is empty.')
    assert isinstance(code, str)
    try:
        self.ast = mparser.Parser(code, mesonfile).parse()
        self.handle_meson_version_from_ast()
    except mparser.ParseException as me:
        me.file = mesonfile
        if me.ast:
            self.ast = me.ast
            self.handle_meson_version_from_ast()
        raise me