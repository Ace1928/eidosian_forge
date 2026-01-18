import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
def register_tab_comp_context(self, *args, **kwargs):
    """Wrapper around TabCompletionRegistry.register_tab_comp_context()."""
    self._tab_completion_registry.register_tab_comp_context(*args, **kwargs)