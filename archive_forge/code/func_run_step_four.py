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
def run_step_four():
    """
    Function for running the fourth step
    """
    config.show_env(f'Step 4: {config.app_name} Installation & Validation')
    os.system(f'pip install {APP_PATH}')
    Path('/data').mkdir(exist_ok=True)
    add_to_env(f'{config.app_name.upper()}_DATA_DIR', '/data')
    import time
    add_to_env(f'{config.app_name.upper()}_BUILD_DATE', int(time.time()))
    for t in config.enabled_build_services:
        names = config.builds[t].get('names', [t])
        for name in names:
            if config.has_service(name):
                run_validate_step(name, fixed=name == 'server')
    echo(f'{COLOR.GREEN}Step 4: {config.app_name} Validation Complete{COLOR.END}\n\n')