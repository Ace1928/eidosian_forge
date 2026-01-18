import argparse
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
from cmdstanpy.utils.cmdstan import get_download_url
from . import progress as progbar
@cached_property
def overwrite(self) -> bool:
    print('Overwrite existing CmdStan installation?')
    print('Default: n')
    answer = input('[y/n]: ')
    return yes_no(answer, False)