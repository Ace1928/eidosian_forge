import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
def set_help_intro(self, help_intro):
    """Set an introductory message to the help output of the command registry.

    Args:
      help_intro: (RichTextLines) Rich text lines appended to the beginning of
        the output of the command "help", as introductory information.
    """
    self._command_handler_registry.set_help_intro(help_intro=help_intro)