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
def run_step_two():
    """
    Function for installing stage two requirements
    """
    config.show_env(f'Step 2: {config.app_name} Pip Installations')
    service_names = config.enabled_build_services
    for service in service_names:
        kind = config.builds[service]['kind']
        echo(f'{COLOR.BLUE}[{kind}]{COLOR.END} Installing {COLOR.BOLD}{service}{COLOR.END} requirements')
        if (req_file := get_pip_requirements(kind, service)):
            os.system(f'pip install -r {req_file}')
            if 'custom_commands' in config.builds[service]:
                for custom in config.builds[service]['custom_commands']:
                    if (custom_cmd := config.custom_commands.get(custom)):
                        echo(f'{COLOR.BLUE}[{kind}]{COLOR.END} Running {COLOR.BOLD}{custom}{COLOR.END}')
                        for cmdstr in custom_cmd:
                            os.system(cmdstr)
    echo(f'{COLOR.GREEN}Step 2: {config.app_name} Pip Installations Complete{COLOR.END}\n\n')