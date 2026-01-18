from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def cmake_get_generator_args(env: 'Environment') -> T.List[str]:
    backend_name = env.coredata.get_option(OptionKey('backend'))
    assert isinstance(backend_name, str)
    assert backend_name in backend_generator_map
    return ['-G', backend_generator_map[backend_name]]