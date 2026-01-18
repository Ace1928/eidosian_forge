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
def run_apt_install():
    """
    Helper for running apt install
    """
    if not APT_PKGS:
        return
    _pkgs = ' '.join(list(set(APT_PKGS)))
    echo(f'Installing Apt Packages: {COLOR.BLUE}{_pkgs}{COLOR.END}')
    os.system(f'apt-get update && apt-get -yq install --no-install-recommends {_pkgs}')