import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
def is_untracked(self, file_name: str) -> Optional[bool]:
    if not self.repo:
        return True
    try:
        return file_name in self.repo.untracked_files
    except GitCommandError:
        return None