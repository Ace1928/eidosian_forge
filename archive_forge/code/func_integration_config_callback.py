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
def integration_config_callback(payload_config: PayloadConfig) -> None:
    """
            Add the integration config vars file to the payload file list.
            This will preserve the file during delegation even if the file is ignored by source control.
            """
    files = payload_config.files
    files.append((vars_file_src, data_context().content.integration_vars_path))