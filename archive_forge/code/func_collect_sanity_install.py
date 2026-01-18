from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def collect_sanity_install(sanity: str) -> list[PipInstall]:
    """Return the details necessary for the specified sanity pip install(s)."""
    requirements_paths: list[tuple[str, str]] = []
    constraints_paths: list[tuple[str, str]] = []
    path = os.path.join(ANSIBLE_TEST_DATA_ROOT, 'requirements', f'sanity.{sanity}.txt')
    requirements_paths.append((ANSIBLE_TEST_DATA_ROOT, path))
    if data_context().content.is_ansible:
        path = os.path.join(data_context().content.sanity_path, 'code-smell', f'{sanity}.requirements.txt')
        requirements_paths.append((data_context().content.root, path))
    return collect_install(requirements_paths, constraints_paths, constraints=False)