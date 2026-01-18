import os
import platform
import shutil
import sys
import ctypes
from time import monotonic as clock
import configparser
from typing import Union
from .. import sparse
from .. import constants as const
import logging
import subprocess
from uuid import uuid4
def silent_remove(self, file: Union[str, bytes, os.PathLike]) -> None:
    try:
        os.remove(file)
    except FileNotFoundError:
        pass