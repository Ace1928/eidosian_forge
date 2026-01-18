import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def resolve_repository_name(repo_name: str) -> Tuple[str, str]:
    if '://' in repo_name:
        raise InvalidRepositoryError(f'Repository name cannot contain a scheme ({repo_name})')
    index_name, remote_name = split_repo_name(repo_name)
    if index_name[0] == '-' or index_name[-1] == '-':
        raise InvalidRepositoryError(f'Invalid index name ({index_name}). Cannot begin or end with a hyphen.')
    return (resolve_index_name(index_name), remote_name)