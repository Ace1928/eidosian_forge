import importlib.util
import os
import platform
from argparse import ArgumentParser
import huggingface_hub
from .. import __version__ as version
from ..utils import (
from . import BaseTransformersCLICommand
@staticmethod
def format_dict(d):
    return '\n'.join([f'- {prop}: {val}' for prop, val in d.items()]) + '\n'