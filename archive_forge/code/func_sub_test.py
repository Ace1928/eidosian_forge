import sys
import pygame
import pygame.threads
import os
import re
import shutil
import tempfile
import time
import random
from pprint import pformat
def sub_test(module):
    print(f'loading {module}')
    cmd = [option_python, '-m', test_runner_mod, module] + pass_on_args
    return (module, (cmd, test_env, working_dir), proc_in_time_or_kill(cmd, option_time_out, env=test_env, wd=working_dir))