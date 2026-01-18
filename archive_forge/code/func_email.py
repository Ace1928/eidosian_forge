import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
@property
def email(self) -> Optional[str]:
    if not self.repo:
        return None
    try:
        return self.repo.config_reader().get_value('user', 'email')
    except configparser.Error:
        return None