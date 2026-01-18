import os
import yaml
import typer
import contextlib
import subprocess
import concurrent.futures
from pathlib import Path
from pydantic import model_validator
from lazyops.types.models import BaseModel
from lazyops.libs.proxyobj import ProxyObject
from typing import Optional, List, Any, Dict, Union
def get_apt_packages(kind: str, name: str) -> List[str]:
    """
    Helper for getting the apt packages for a given service/builder
    """
    pkg_dir = PKGS_PATH
    for alias in config.kinds[kind].get(name, []):
        pkg_file = pkg_dir.joinpath(f'{alias}.txt')
        if pkg_file.exists():
            return parse_text_file(pkg_file)
    return []