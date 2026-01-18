import functools
import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import traceback
from typing import Dict
from ..runtime.driver import DriverBase
def make_launcher_stub(self, name, signature, constants):
    """
        Generate the launcher stub to launch the kernel
        """
    raise NotImplementedError