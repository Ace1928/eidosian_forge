from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def cmake_is_debug(env: 'Environment') -> bool:
    if OptionKey('b_vscrt') in env.coredata.options:
        is_debug = env.coredata.get_option(OptionKey('buildtype')) == 'debug'
        if env.coredata.options[OptionKey('b_vscrt')].value in {'mdd', 'mtd'}:
            is_debug = True
        return is_debug
    else:
        debug_opt = env.coredata.get_option(OptionKey('debug'))
        assert isinstance(debug_opt, bool)
        return debug_opt