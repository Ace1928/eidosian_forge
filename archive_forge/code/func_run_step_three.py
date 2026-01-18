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
def run_step_three():
    """
    Function for running the third step
    """
    config.show_env(f'Step 3: {config.app_name} Post-Installations')
    init_build_deps()
    echo(f'{COLOR.GREEN}Step 3: {config.app_name} Post-Installations Complete{COLOR.END}\n\n')