import contextlib
import os
import platform
import re
import shutil
import tempfile
from typing import Any, Iterator, List, Mapping, Optional, Tuple, Union
from cmdstanpy import _TMPDIR
from .json import write_stan_json
from .logging import get_logger
@contextlib.contextmanager
def temp_inits(inits: Union[str, os.PathLike, Mapping[str, Any], float, int, List[Any], None], *, allow_multiple: bool=True, id: int=1) -> Iterator[Union[str, float, int, None]]:
    if isinstance(inits, (float, int)):
        yield inits
        return
    if allow_multiple:
        yield from _temp_multiinput(inits, base=id)
    else:
        if isinstance(inits, list):
            raise ValueError('Expected single initialization, got list')
        yield from _temp_single_json(inits)