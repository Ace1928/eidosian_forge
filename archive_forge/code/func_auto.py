import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
@property
def auto(self) -> bool:
    return self._remote_url is None