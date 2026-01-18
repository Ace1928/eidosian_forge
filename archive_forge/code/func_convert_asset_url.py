import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import requests
import typer
from wasabi import msg
from ..util import SimpleFrozenDict, download_file, ensure_path, get_checksum
from ..util import get_git_version, git_checkout, load_project_config
from ..util import parse_config_overrides, working_dir
from .main import PROJECT_FILE, Arg, Opt, app
def convert_asset_url(url: str) -> str:
    """Check and convert the asset URL if needed.

    url (str): The asset URL.
    RETURNS (str): The converted URL.
    """
    if re.match('(http(s?)):\\/\\/github.com', url) and 'releases/download' not in url and ('/raw/' not in url):
        converted = url.replace('github.com', 'raw.githubusercontent.com')
        converted = re.sub('/(tree|blob)/', '/', converted)
        msg.warn('Downloading from a regular GitHub URL. This will only download the source of the page, not the actual file. Converting the URL to a raw URL.', converted)
        return converted
    return url