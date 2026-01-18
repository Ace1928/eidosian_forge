import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from multiprocessing import cpu_count
from typing import (
import pandas as pd
from tqdm.auto import tqdm
from cmdstanpy import (
from cmdstanpy.cmdstan_args import (
from cmdstanpy.stanfit import (
from cmdstanpy.utils import (
from cmdstanpy.utils.filesystem import temp_inits, temp_single_json
from . import progress as progbar
def src_info(self) -> Dict[str, Any]:
    """
        Run stanc with option '--info'.

        If stanc is older than 2.27 or if the stan
        file cannot be found, returns an empty dictionary.
        """
    if self.stan_file is None or cmdstan_version_before(2, 27):
        return {}
    return compilation.src_info(str(self.stan_file), self._compiler_options)