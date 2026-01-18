from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
def get_ansible_config(self) -> str:
    """Return the path to the Ansible config for the given config."""
    ansible_config_relative_path = os.path.join(data_context().content.integration_path, '%s.cfg' % self.command)
    ansible_config_path = os.path.join(data_context().content.root, ansible_config_relative_path)
    if not os.path.exists(ansible_config_path):
        ansible_config_path = super().get_ansible_config()
    return ansible_config_path