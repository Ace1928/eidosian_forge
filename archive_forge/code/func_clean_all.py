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
def clean_all(verbose: bool=False) -> None:
    """
    Run `make clean-all` in the current directory (must be a cmdstan library).

    :param verbose: Boolean value; when ``True``, show output from make command.
    """
    cmd = [MAKE, 'clean-all']
    try:
        if verbose:
            do_command(cmd)
        else:
            do_command(cmd, fd_out=None)
    except RuntimeError as e:
        raise CmdStanInstallError(f'Command "make clean-all" failed\n{str(e)}')