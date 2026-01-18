import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def monitored(self, args, unknown=None):
    global _args_argparse
    _args_argparse = copy.deepcopy(vars(args))