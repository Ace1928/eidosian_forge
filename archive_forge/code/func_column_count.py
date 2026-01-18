import argparse
import os
import re
import numpy as np
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import profiling
from tensorflow.python.debug.lib import source_utils
def column_count(self):
    return len(self._column_names)