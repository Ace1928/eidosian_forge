import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def split_repo_name(repo_name: str) -> Tuple[str, str]:
    parts = repo_name.split('/', 1)
    if len(parts) == 1 or ('.' not in parts[0] and ':' not in parts[0] and (parts[0] != 'localhost')):
        return (INDEX_NAME, repo_name)
    return (parts[0], parts[1])