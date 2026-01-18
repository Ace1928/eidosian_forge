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
def list_profile(self, args, screen_info=None):
    """Command handler for list_profile.

    List per-operation profile information.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    screen_cols = 80
    if screen_info and 'cols' in screen_info:
        screen_cols = screen_info['cols']
    parsed = self._arg_parsers['list_profile'].parse_args(args)
    op_time_interval = command_parser.parse_time_interval(parsed.op_time) if parsed.op_time else None
    exec_time_interval = command_parser.parse_time_interval(parsed.execution_time) if parsed.execution_time else None
    node_name_regex = re.compile(parsed.node_name_filter) if parsed.node_name_filter else None
    file_path_regex = re.compile(parsed.file_path_filter) if parsed.file_path_filter else None
    op_type_regex = re.compile(parsed.op_type_filter) if parsed.op_type_filter else None
    output = debugger_cli_common.RichTextLines([''])
    device_name_regex = re.compile(parsed.device_name_filter) if parsed.device_name_filter else None
    data_generator = self._get_profile_data_generator()
    device_count = len(self._run_metadata.step_stats.dev_stats)
    for index in range(device_count):
        device_stats = self._run_metadata.step_stats.dev_stats[index]
        if not device_name_regex or device_name_regex.match(device_stats.device):
            profile_data = [datum for datum in data_generator(device_stats) if _list_profile_filter(datum, node_name_regex, file_path_regex, op_type_regex, op_time_interval, exec_time_interval, min_lineno=parsed.min_lineno, max_lineno=parsed.max_lineno)]
            profile_data = sorted(profile_data, key=lambda datum: _list_profile_sort_key(datum, parsed.sort_by), reverse=parsed.reverse)
            output.extend(self._get_list_profile_lines(device_stats.device, index, device_count, profile_data, parsed.sort_by, parsed.reverse, parsed.time_unit, device_name_filter=parsed.device_name_filter, node_name_filter=parsed.node_name_filter, op_type_filter=parsed.op_type_filter, screen_cols=screen_cols))
    return output