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
def info_model_url(model: str) -> Dict[str, Any]:
    """Return the download URL for the latest version of a pipeline."""
    version = get_latest_version(model)
    filename = get_model_filename(model, version)
    download_url = about.__download_url__ + '/' + filename
    release_tpl = 'https://github.com/explosion/spacy-models/releases/tag/{m}-{v}'
    release_url = release_tpl.format(m=model, v=version)
    return {'download_url': download_url, 'release_url': release_url}