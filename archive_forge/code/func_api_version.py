import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
@property
def api_version(self) -> str:
    """The API version used (4 only)."""
    return self._api_version