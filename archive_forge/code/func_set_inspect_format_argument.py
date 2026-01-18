import json
import pathlib
import warnings
from typing import IO, Union, Optional, Literal
from .mimebundle import spec_to_mimebundle
from ..vegalite.v5.data import data_transformers
from altair.utils._vegafusion_data import using_vegafusion
def set_inspect_format_argument(format: Optional[str], fp: Union[str, pathlib.PurePath, IO], inline: bool) -> str:
    """Inspect the format argument in the save function"""
    if format is None:
        if isinstance(fp, str):
            format = fp.split('.')[-1]
        elif isinstance(fp, pathlib.PurePath):
            format = fp.suffix.lstrip('.')
        else:
            raise ValueError("must specify file format: ['png', 'svg', 'pdf', 'html', 'json', 'vega']")
    if format != 'html' and inline:
        warnings.warn('inline argument ignored for non HTML formats.', stacklevel=1)
    return format