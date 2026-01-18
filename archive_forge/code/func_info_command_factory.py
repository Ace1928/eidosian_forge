import importlib.util
import os
import platform
from argparse import ArgumentParser
import huggingface_hub
from .. import __version__ as version
from ..utils import (
from . import BaseTransformersCLICommand
def info_command_factory(_):
    return EnvironmentCommand()