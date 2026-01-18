from __future__ import annotations
import abc
import datetime
import os
import re
import tempfile
import time
import typing as t
from ....encoding import (
from ....io import (
from ....util import (
from ....util_common import (
from ....target import (
from ....config import (
from ....ci import (
from ....data import (
from ....docker_util import (
def config_callback(payload_config: PayloadConfig) -> None:
    """Add the config file to the payload file list."""
    if self.platform not in self.args.metadata.cloud_config:
        return
    if self._get_cloud_config(self._CONFIG_PATH, ''):
        pair = (self.config_path, os.path.relpath(self.config_path, data_context().content.root))
        files = payload_config.files
        if pair not in files:
            display.info('Including %s config: %s -> %s' % (self.platform, pair[0], pair[1]), verbosity=3)
            files.append(pair)