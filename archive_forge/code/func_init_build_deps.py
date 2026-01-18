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
def init_build_deps():
    """
    Initialize the worker dependencies

    - Do this multi-threaded later
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(init_build_service, name) for name in config.enabled_build_services]
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()