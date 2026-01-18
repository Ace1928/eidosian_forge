from __future__ import annotations
import ast
import base64
import builtins  # Explicitly use builtins.set as 'set' will be shadowed by a function
import json
import os
import pathlib
import site
import sys
import threading
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, overload
import yaml
from dask.typing import no_default
def collect_yaml(paths: Sequence[str], *, return_paths: bool=False) -> Iterator[dict | tuple[pathlib.Path, dict]]:
    """Collect configuration from yaml files

    This searches through a list of paths, expands to find all yaml or json
    files, and then parses each file.
    """
    file_paths = []
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    file_paths.extend(sorted((os.path.join(path, p) for p in os.listdir(path) if os.path.splitext(p)[1].lower() in ('.json', '.yaml', '.yml'))))
                except OSError:
                    pass
            else:
                file_paths.append(path)
    for path in file_paths:
        config = _load_config_file(path)
        if config is not None:
            if return_paths:
                yield (pathlib.Path(path), config)
            else:
                yield config