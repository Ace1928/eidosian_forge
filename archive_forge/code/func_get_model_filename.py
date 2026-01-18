import sys
from typing import Optional, Sequence
import requests
import typer
from wasabi import msg
from .. import about
from ..errors import OLD_MODEL_SHORTCUTS
from ..util import (
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app
def get_model_filename(model_name: str, version: str, sdist: bool=False) -> str:
    dl_tpl = '{m}-{v}/{m}-{v}{s}'
    suffix = SDIST_SUFFIX if sdist else WHEEL_SUFFIX
    filename = dl_tpl.format(m=model_name, v=version, s=suffix)
    return filename