import argparse
from dataclasses import dataclass
from enum import Enum
import os.path
import tempfile
import typer
from typing import Optional
import requests
from ray.tune.experiment.config_parser import _make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
def download_example_file(example_file: str, base_url: Optional[str]='https://raw.githubusercontent.com/' + 'ray-project/ray/master/rllib/'):
    """Download the example file (e.g. from GitHub) if it doesn't exist locally.
    If the provided example file exists locally, we return it directly.

    Not every user will have cloned our repo and cd'ed into this working directory
    when using the CLI.

    Args:
        example_file: The example file to download.
        base_url: The base URL to download the example file from. Use this if
            'example_file' is a link relative to this base URL. If set to 'None',
            'example_file' is assumed to be a complete URL (or a local file, in which
            case nothing is downloaded).
    """
    temp_file = None
    if not os.path.exists(example_file):
        example_url = base_url + example_file if base_url else example_file
        print(f'>>> Attempting to download example file {example_url}...')
        file_type = get_file_type(example_url)
        if file_type == SupportedFileType.yaml:
            temp_file = tempfile.NamedTemporaryFile(suffix='.yaml')
        else:
            assert file_type == SupportedFileType.python, f'`example_url` ({example_url}) must be a python or yaml file!'
            temp_file = tempfile.NamedTemporaryFile(suffix='.py')
        r = requests.get(example_url)
        with open(temp_file.name, 'wb') as f:
            print(r.content)
            f.write(r.content)
        print(f'  Status code: {r.status_code}')
        if r.status_code == 200:
            print(f'  Downloaded example file to {temp_file.name}')
            example_file = temp_file.name
    return (example_file, temp_file)