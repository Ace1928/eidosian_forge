import json
import os
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List
from ..utils import logging
from . import BaseTransformersCLICommand
def skip_units(line):
    return 'generating PyTorch' in line and (not output_pytorch) or ('generating TensorFlow' in line and (not output_tensorflow)) or ('generating Flax' in line and (not output_flax))