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
def build_aliases(name: str, additional_names: Optional[List[str]]=None) -> List[str]:
    """
    Create the aliases for a given service name
    """
    aliases = [name]
    if '.' not in name:
        if '-' in name:
            aliases.append(name.replace('-', '.'))
        elif '_' in name:
            aliases.append(name.replace('_', '.'))
    if '-' not in name:
        if '.' in name:
            aliases.append(name.replace('.', '-'))
        elif '_' in name:
            aliases.append(name.replace('_', '-'))
    if '_' not in name:
        if '.' in name:
            aliases.append(name.replace('.', '_'))
        elif '-' in name:
            aliases.append(name.replace('-', '_'))
    if additional_names:
        aliases.extend(additional_names)
    return list(set(aliases))