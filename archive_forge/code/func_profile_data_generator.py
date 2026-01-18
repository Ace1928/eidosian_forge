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
def profile_data_generator(device_step_stats):
    for node_stats in device_step_stats.node_stats:
        if node_stats.node_name == '_SOURCE' or node_stats.node_name == '_SINK':
            continue
        yield profiling.ProfileDatum(device_step_stats.device, node_stats, node_to_file_path.get(node_stats.node_name, ''), node_to_line_number.get(node_stats.node_name, 0), node_to_func_name.get(node_stats.node_name, ''), node_to_op_type.get(node_stats.node_name, ''))