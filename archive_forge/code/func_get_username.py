import configparser
import getpass
import os
import tempfile
from typing import Any, Optional
from wandb import env
from wandb.old import core
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.runid import generate_id
def get_username() -> str:
    try:
        return getpass.getuser()
    except (ImportError, KeyError):
        return generate_id()