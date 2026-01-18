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
def get_apt_pkg_requirements():
    """
    Helper for getting the apt package requirements
    """
    for service in config.builds:
        if config.has_service(service):
            if (service_pkgs := get_apt_packages(config.builds[service]['kind'], service)):
                echo(f'{COLOR.BLUE}[{config.builds[service]['kind']}]{COLOR.END} Adding {COLOR.BOLD}{service}{COLOR.END} requirements\n\n - {COLOR.BOLD}{service_pkgs}{COLOR.END}\n')
                add_to_apt_pkgs(*service_pkgs)