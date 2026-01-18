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
def info_installed_model_url(model: str) -> Optional[str]:
    """Given a pipeline name, get the download URL if available, otherwise
    return None.

    This is only available for pipelines installed as modules that have
    dist-info available.
    """
    try:
        dist = importlib_metadata.distribution(model)
        text = dist.read_text('direct_url.json')
        if isinstance(text, str):
            data = json.loads(text)
            return data['url']
    except Exception:
        pass
    return None