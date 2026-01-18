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
@cmd.command('breakcache')
def run_break_cache():
    """
    Runs the break cache command
    """
    os.system('echo "Breaking Cache: $(date)" > /tmp/.breakcache')