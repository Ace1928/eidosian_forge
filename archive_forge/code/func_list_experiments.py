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
def list_experiments(project_path: str, sort: Optional[List[str]]=None, output: str=None, filter_op: str=None, info_keys: Optional[List[str]]=None, limit: int=None, desc: bool=False):
    """Lists experiments in the directory subtree.

    Args:
        project_path: Directory where experiments are located.
            Corresponds to Experiment.local_dir.
        sort: Keys to sort by.
        output: Name of file where output is saved.
        filter_op: Filter operation in the format
            "<column> <operator> <value>".
        info_keys: Keys that are displayed.
        limit: Number of rows to display.
        desc: Sort ascending vs. descending.
    """
    _check_tabulate()
    base, experiment_folders, _ = next(os.walk(project_path))
    experiment_data_collection = []
    for experiment_dir in experiment_folders:
        num_trials = sum((EXPR_RESULT_FILE in files for _, _, files in os.walk(os.path.join(base, experiment_dir))))
        experiment_data = {'name': experiment_dir, 'total_trials': num_trials}
        experiment_data_collection.append(experiment_data)
    if not experiment_data_collection:
        raise click.ClickException('No experiments found!')
    info_df = pd.DataFrame(experiment_data_collection)
    if not info_keys:
        info_keys = DEFAULT_PROJECT_INFO_KEYS
    col_keys = [k for k in list(info_keys) if k in info_df]
    if not col_keys:
        raise click.ClickException('None of keys {} in experiment data!'.format(info_keys))
    info_df = info_df[col_keys]
    if filter_op:
        col, op, val = filter_op.split(' ')
        col_type = info_df[col].dtype
        if is_numeric_dtype(col_type):
            val = float(val)
        elif is_string_dtype(col_type):
            val = str(val)
        else:
            raise click.ClickException('Unsupported dtype for {}: {}'.format(val, col_type))
        op = OPERATORS[op]
        filtered_index = op(info_df[col], val)
        info_df = info_df[filtered_index]
    if sort:
        for key in sort:
            if key not in info_df:
                raise click.ClickException('{} not in: {}'.format(key, list(info_df)))
        ascending = not desc
        info_df = info_df.sort_values(by=sort, ascending=ascending)
    if limit:
        info_df = info_df[:limit]
    print_format_output(info_df)
    if output:
        file_extension = os.path.splitext(output)[1].lower()
        if file_extension in ('.p', '.pkl', '.pickle'):
            info_df.to_pickle(output)
        elif file_extension == '.csv':
            info_df.to_csv(output, index=False)
        else:
            raise click.ClickException('Unsupported filetype: {}'.format(output))
        click.secho('Output saved at {}'.format(output), fg='green')