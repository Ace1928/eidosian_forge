from pathlib import Path
from typing import Optional, List
import click
import logging
import operator
import os
import shutil
import subprocess
from datetime import datetime
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from ray.air.constants import EXPR_RESULT_FILE
from ray.tune.result import (
from ray.tune.analysis import ExperimentAnalysis
from ray.tune import TuneError
from ray._private.thirdparty.tabulate.tabulate import tabulate
def add_note(path: str, filename: str='note.txt'):
    """Opens a txt file at the given path where user can add and save notes.

    Args:
        path: Directory where note will be saved.
        filename: Name of note. Defaults to "note.txt"
    """
    path = Path(path).expanduser()
    assert path.is_dir(), '{} is not a valid directory.'.format(path)
    filepath = path / filename
    try:
        subprocess.call([EDITOR, filepath.as_posix()])
    except Exception as exc:
        click.secho('Editing note failed: {}'.format(str(exc)), fg='red')
    if filepath.exists():
        print('Note updated at:', filepath.as_posix())
    else:
        print('Note created at:', filepath.as_posix())