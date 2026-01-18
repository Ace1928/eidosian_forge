import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import srsly
from catalogue import RegistryError
from thinc.api import Config
from wasabi import MarkdownRenderer, Printer, get_raw_input
from .. import about, util
from ..compat import importlib_metadata
from ..schemas import ModelMetaSchema, validate
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app, string_to_list
import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
def get_build_formats(formats: List[str]) -> Tuple[bool, bool]:
    supported = ['sdist', 'wheel', 'none']
    for form in formats:
        if form not in supported:
            msg = Printer()
            err = f'Unknown build format: {form}. Supported: {', '.join(supported)}'
            msg.fail(err, exits=1)
    if not formats or 'none' in formats:
        return (False, False)
    return ('sdist' in formats, 'wheel' in formats)