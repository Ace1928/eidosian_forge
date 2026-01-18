import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from collections import OrderedDict
from time import sleep
from typing import Any, Dict, List
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import pushd, validate_dir, wrap_url_progress_hook
def retrieve_toolchain(filename: str, url: str, progress: bool=True) -> None:
    """Download toolchain from URL."""
    print('Downloading C++ toolchain: {}'.format(filename))
    for i in range(6):
        try:
            if progress:
                progress_hook = wrap_url_progress_hook()
            else:
                progress_hook = None
            _ = urllib.request.urlretrieve(url, filename=filename, reporthook=progress_hook)
            break
        except urllib.error.URLError as err:
            print('Failed to download C++ toolchain')
            print(err)
            if i < 5:
                print('retry ({}/5)'.format(i + 1))
                sleep(1)
                continue
            sys.exit(3)
    print('Download successful, file: {}'.format(filename))