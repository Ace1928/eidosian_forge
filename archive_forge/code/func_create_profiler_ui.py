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
def create_profiler_ui(graph, run_metadata, ui_type='readline', on_ui_exit=None, config=None):
    """Create an instance of ReadlineUI based on a `tf.Graph` and `RunMetadata`.

  Args:
    graph: Python `Graph` object.
    run_metadata: A `RunMetadata` protobuf object.
    ui_type: (str) requested UI type, e.g., "readline".
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.
    config: An instance of `cli_config.CLIConfig`.

  Returns:
    (base_ui.BaseUI) A BaseUI subtype object with a set of standard analyzer
      commands and tab-completions registered.
  """
    del config
    analyzer = ProfileAnalyzer(graph, run_metadata)
    cli = ui_factory.get_ui(ui_type, on_ui_exit=on_ui_exit)
    cli.register_command_handler('list_profile', analyzer.list_profile, analyzer.get_help('list_profile'), prefix_aliases=['lp'])
    cli.register_command_handler('print_source', analyzer.print_source, analyzer.get_help('print_source'), prefix_aliases=['ps'])
    return cli