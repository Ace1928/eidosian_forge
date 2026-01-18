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
def add_world_args(self, task, interactive_task, selfchat_task):
    """
        Add arguments specific to the world.
        """
    world_class = load_world_module(task, interactive_task=interactive_task, selfchat_task=selfchat_task)
    if world_class is not None and hasattr(world_class, 'add_cmdline_args'):
        try:
            world_class.add_cmdline_args(self)
        except argparse.ArgumentError:
            pass