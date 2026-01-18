from __future__ import annotations
import collections.abc as c
import contextlib
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import typing as t
from ...encoding import (
from ...ansible_util import (
from ...executor import (
from ...python_requirements import (
from ...ci import (
from ...target import (
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...cache import (
from .cloud import (
from ...data import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...inventory import (
from .filters import (
from .coverage import (
@contextlib.contextmanager
def integration_test_config_file(args: IntegrationConfig, env_config: CloudEnvironmentConfig, integration_dir: str) -> c.Iterator[t.Optional[str]]:
    """Context manager that provides a config file for integration tests, if needed."""
    if not env_config:
        yield None
        return
    config_vars = (env_config.ansible_vars or {}).copy()
    config_vars.update(ansible_test=dict(environment=env_config.env_vars, module_defaults=env_config.module_defaults))
    config_file = json.dumps(config_vars, indent=4, sort_keys=True)
    with named_temporary_file(args, 'config-file-', '.json', integration_dir, config_file) as path:
        filename = os.path.relpath(path, integration_dir)
        display.info('>>> Config File: %s\n%s' % (filename, config_file), verbosity=3)
        yield path