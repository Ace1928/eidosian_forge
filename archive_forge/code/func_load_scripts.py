import copy
import re
import threading
import time
import warnings
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union
from redis._parsers.encoders import Encoder
from redis._parsers.helpers import (
from redis.commands import (
from redis.connection import (
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def load_scripts(self):
    scripts = list(self.scripts)
    immediate = self.immediate_execute_command
    shas = [s.sha for s in scripts]
    exists = immediate('SCRIPT EXISTS', *shas)
    if not all(exists):
        for s, exist in zip(scripts, exists):
            if not exist:
                s.sha = immediate('SCRIPT LOAD', s.script)