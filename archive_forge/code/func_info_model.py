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
def info_model(model: str, *, silent: bool=True) -> Dict[str, Any]:
    """Generate info about a specific model.

    model (str): Model name of path.
    silent (bool): Don't print anything, just return.
    RETURNS (dict): The model meta.
    """
    msg = Printer(no_print=silent, pretty=not silent)
    if util.is_package(model):
        model_path = util.get_package_path(model)
    else:
        model_path = Path(model)
    meta_path = model_path / 'meta.json'
    if not meta_path.is_file():
        msg.fail("Can't find pipeline meta.json", meta_path, exits=1)
    meta = srsly.read_json(meta_path)
    if model_path.resolve() != model_path:
        meta['source'] = str(model_path.resolve())
    else:
        meta['source'] = str(model_path)
    download_url = info_installed_model_url(model)
    if download_url:
        meta['download_url'] = download_url
    return {k: v for k, v in meta.items() if k not in ('accuracy', 'performance', 'speed')}