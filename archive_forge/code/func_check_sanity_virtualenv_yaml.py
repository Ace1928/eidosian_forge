from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
def check_sanity_virtualenv_yaml(python: VirtualPythonConfig) -> t.Optional[bool]:
    """Return True if PyYAML has libyaml support for the given sanity virtual environment, False if it does not and None if it was not found."""
    virtualenv_path = os.path.dirname(os.path.dirname(python.path))
    meta_yaml = os.path.join(virtualenv_path, 'meta.yaml.json')
    virtualenv_yaml = read_json_file(meta_yaml)
    return virtualenv_yaml