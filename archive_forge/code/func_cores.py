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
def cores(self) -> int:
    max_cpus = os.cpu_count() or 1
    print('How many CPU cores would you like to use for installing and compiling CmdStan?')
    print(f'Default: 1, Max: {max_cpus}')
    answer = input('Enter a number or hit enter to continue: ')
    try:
        return min(max_cpus, max(int(answer), 1))
    except ValueError:
        return 1