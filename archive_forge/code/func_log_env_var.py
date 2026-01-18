import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
def log_env_var(self, env_var_name=''):
    if env_var_name in os.environ:
        logger.info('%s=%s', env_var_name, os.environ[env_var_name])