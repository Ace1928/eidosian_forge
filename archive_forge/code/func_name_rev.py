import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
@property
def name_rev(self) -> str:
    """
        :return:
            String describing the commits hex sha based on the closest Reference.
            Mostly useful for UI purposes
        """
    return self.repo.git.name_rev(self)