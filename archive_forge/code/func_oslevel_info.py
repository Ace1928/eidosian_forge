import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
def oslevel_info(self) -> str:
    """
        Return AIX' oslevel command output.
        """
    return self._oslevel_info