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
def add_parlai_data_path(self, argument_group=None):
    """
        Add --datapath CLI arg.
        """
    if argument_group is None:
        argument_group = self
    argument_group.add_argument('-dp', '--datapath', default=None, help='path to datasets, defaults to {parlai_dir}/data')