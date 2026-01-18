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
def show_env(self, step: str):
    """
        Show the environment
        """
    echo(f'Starting Step: {COLOR.GREEN}{step}{COLOR.END}\n')
    echo(f'[Enabled Builds]: {COLOR.BLUE}{self.enabled_build_services}{COLOR.END}')