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
def add_model_subargs(self, model):
    """
        Add arguments specific to a particular model.
        """
    agent = load_agent_module(model)
    try:
        if hasattr(agent, 'add_cmdline_args'):
            agent.add_cmdline_args(self)
    except argparse.ArgumentError:
        pass
    try:
        if hasattr(agent, 'dictionary_class'):
            s = class2str(agent.dictionary_class())
            self.set_defaults(dict_class=s)
    except argparse.ArgumentError:
        pass