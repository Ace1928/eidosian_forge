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
def add_task_args(self, task):
    """
        Add arguments specific to the specified task.
        """
    for t in ids_to_tasks(task).split(','):
        agent = load_teacher_module(t)
        try:
            if hasattr(agent, 'add_cmdline_args'):
                agent.add_cmdline_args(self)
        except argparse.ArgumentError:
            pass