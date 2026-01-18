from __future__ import annotations
import json
import os
import shutil
import typing as t
from .constants import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .python_requirements import (
from .host_configs import (
from .thread import (
def get_collection_detail(python: PythonConfig) -> CollectionDetail:
    """Return collection detail."""
    collection = data_context().content.collection
    directory = os.path.join(collection.root, collection.directory)
    stdout = raw_command([python.path, os.path.join(ANSIBLE_TEST_TOOLS_ROOT, 'collection_detail.py'), directory], capture=True)[0]
    result = json.loads(stdout)
    error = result.get('error')
    if error:
        raise CollectionDetailError(error)
    version = result.get('version')
    detail = CollectionDetail()
    detail.version = str(version) if version is not None else None
    return detail