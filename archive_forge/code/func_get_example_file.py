import collections
from rich.console import Console
from rich.table import Table
import typer
from ray.rllib import train as train_module
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import (
def get_example_file(example_id):
    """Simple helper function to get the example file for a given example ID."""
    if example_id not in EXAMPLES:
        raise example_error(example_id)
    example = EXAMPLES[example_id]
    assert 'file' in example.keys(), f"Example {example_id} does not have a 'file' attribute."
    return example.get('file')