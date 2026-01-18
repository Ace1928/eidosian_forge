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
def build_number(self, best: bool=False) -> str:
    """
        Return the build number of the current distribution.

        For details, see :func:`distro.build_number`.
        """
    return self.version_parts(best)[2]