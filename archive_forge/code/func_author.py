from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
@classmethod
def author(cls, config_reader: Union[None, 'GitConfigParser', 'SectionConstraint']=None) -> 'Actor':
    """Same as committer(), but defines the main author. It may be specified in the
        environment, but defaults to the committer."""
    return cls._main_actor(cls.env_author_name, cls.env_author_email, config_reader)