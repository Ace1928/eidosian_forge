import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
def run_ui(self, init_command=None, title=None, title_color=None, enable_mouse_on_start=True):
    """Run the UI until user- or command- triggered exit.

    Args:
      init_command: (str) Optional command to run on CLI start up.
      title: (str) Optional title to display in the CLI.
      title_color: (str) Optional color of the title, e.g., "yellow".
      enable_mouse_on_start: (bool) Whether the mouse mode is to be enabled on
        start-up.

    Returns:
      An exit token of arbitrary type. Can be None.
    """
    raise NotImplementedError('run_ui() is not implemented in BaseUI')