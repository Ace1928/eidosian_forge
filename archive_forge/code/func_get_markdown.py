import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import srsly
from wasabi import MarkdownRenderer, Printer
from .. import about, util
from ..compat import importlib_metadata
from ._util import Arg, Opt, app, string_to_list
from .download import get_latest_version, get_model_filename
def get_markdown(data: Dict[str, Any], title: Optional[str]=None, exclude: Optional[List[str]]=None) -> str:
    """Get data in GitHub-flavoured Markdown format for issues etc.

    data (Dict[str, Any]): Label/value pairs.
    title (str): Optional title, will be rendered as headline 2.
    exclude (List[str]): Names of keys to exclude.
    RETURNS (str): The Markdown string.
    """
    md = MarkdownRenderer()
    if title:
        md.add(md.title(2, title))
    items = []
    for key, value in data.items():
        if exclude and key in exclude:
            continue
        if isinstance(value, str):
            try:
                existing_path = Path(value).exists()
            except Exception:
                existing_path = False
            if existing_path:
                continue
        items.append(f'{md.bold(f'{key}:')} {value}')
    md.add(md.list(items))
    return f'\n{md.text}\n'