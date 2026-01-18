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
def exe_info(self) -> Dict[str, str]:
    """
        Run model with option 'info'. Parse output statements, which all
        have form 'key = value' into a Dict.
        If exe file compiled with CmdStan < 2.27, option 'info' isn't
        available and the method returns an empty dictionary.
        """
    result: Dict[str, str] = {}
    if self.exe_file is None:
        return result
    try:
        info = StringIO()
        do_command(cmd=[str(self.exe_file), 'info'], fd_out=info)
        lines = info.getvalue().split('\n')
        for line in lines:
            kv_pair = [x.strip() for x in line.split('=')]
            if len(kv_pair) != 2:
                continue
            result[kv_pair[0]] = kv_pair[1]
        return result
    except RuntimeError as e:
        get_logger().debug(e)
        return result