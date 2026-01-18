import collections
from rich.console import Console
from rich.table import Table
import typer
from ray.rllib import train as train_module
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import (
def example_error(example_id: str):
    return ValueError(f'Example {example_id} not found. Use `rllib example list` to see available examples.')