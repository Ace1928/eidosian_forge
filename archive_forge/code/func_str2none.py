import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def str2none(value: str):
    """
    If the value is a variant of `none`, return None.

    Otherwise, return the original value.
    """
    if value.lower() == 'none':
        return None
    else:
        return value