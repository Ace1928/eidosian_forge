import argparse
import functools
import os
import pathlib
import re
import sys
import textwrap
from types import ModuleType
from typing import (
from requests.structures import CaseInsensitiveDict
import gitlab.config
from gitlab.base import RESTObject
def docs() -> argparse.ArgumentParser:
    """
    Provide a statically generated parser for sphinx only, so we don't need
    to provide dummy gitlab config for readthedocs.
    """
    if 'sphinx' not in sys.modules:
        sys.exit('Docs parser is only intended for build_sphinx')
    return _get_parser()