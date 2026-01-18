import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
@property
def has_submodule_diff(self) -> bool:
    if not self.repo:
        return False
    return bool(self.repo.git.version_info >= (2, 11, 0))