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
def info_spacy() -> Dict[str, Any]:
    """Generate info about the current spaCy intallation.

    RETURNS (dict): The spaCy info.
    """
    all_models = {}
    for pkg_name in util.get_installed_models():
        package = pkg_name.replace('-', '_')
        all_models[package] = util.get_package_version(pkg_name)
    return {'spaCy version': about.__version__, 'Location': str(Path(__file__).parent.parent), 'Platform': platform.platform(), 'Python version': platform.python_version(), 'Pipelines': all_models}