import json
import os
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List
from ..utils import logging
from . import BaseTransformersCLICommand
def add_new_model_command_factory(args: Namespace):
    return AddNewModelCommand(args.testing, args.testing_file, path=args.path)