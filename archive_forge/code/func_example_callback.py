import collections
from rich.console import Console
from rich.table import Table
import typer
from ray.rllib import train as train_module
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import (
@example_app.callback()
def example_callback():
    """RLlib command-line interface to run built-in examples. You can choose to list
    all available examples, get more information on an example or run a specific
    example.
    """
    pass