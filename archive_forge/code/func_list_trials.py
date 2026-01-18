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
def list_trials(experiment_path: str, sort: Optional[List[str]]=None, output: Optional[str]=None, filter_op: Optional[str]=None, info_keys: Optional[List[str]]=None, limit: int=None, desc: bool=False):
    """Lists trials in the directory subtree starting at the given path.

    Args:
        experiment_path: Directory where trials are located.
            Like Experiment.local_dir/Experiment.name/experiment*.json.
        sort: Keys to sort by.
        output: Name of file where output is saved.
        filter_op: Filter operation in the format
            "<column> <operator> <value>".
        info_keys: Keys that are displayed.
        limit: Number of rows to display.
        desc: Sort ascending vs. descending.
    """
    _check_tabulate()
    try:
        checkpoints_df = ExperimentAnalysis(experiment_path).dataframe()
    except TuneError as e:
        raise click.ClickException('No trial data found!') from e
    config_prefix = CONFIG_PREFIX + '/'

    def key_filter(k):
        return k in DEFAULT_CLI_KEYS or k.startswith(config_prefix)
    col_keys = [k for k in checkpoints_df.columns if key_filter(k)]
    if info_keys:
        for k in info_keys:
            if k not in checkpoints_df.columns:
                raise click.ClickException('Provided key invalid: {}. Available keys: {}.'.format(k, checkpoints_df.columns))
        col_keys = [k for k in checkpoints_df.columns if k in info_keys]
    if not col_keys:
        raise click.ClickException('No columns to output.')
    checkpoints_df = checkpoints_df[col_keys]
    if 'last_update_time' in checkpoints_df:
        with pd.option_context('mode.use_inf_as_null', True):
            datetime_series = checkpoints_df['last_update_time'].dropna()
        datetime_series = datetime_series.apply(lambda t: datetime.fromtimestamp(t).strftime(TIMESTAMP_FORMAT))
        checkpoints_df['last_update_time'] = datetime_series
    if 'logdir' in checkpoints_df:
        checkpoints_df['logdir'] = checkpoints_df['logdir'].str.replace(experiment_path, '')
    if filter_op:
        col, op, val = filter_op.split(' ')
        col_type = checkpoints_df[col].dtype
        if is_numeric_dtype(col_type):
            val = float(val)
        elif is_string_dtype(col_type):
            val = str(val)
        else:
            raise click.ClickException('Unsupported dtype for {}: {}'.format(val, col_type))
        op = OPERATORS[op]
        filtered_index = op(checkpoints_df[col], val)
        checkpoints_df = checkpoints_df[filtered_index]
    if sort:
        for key in sort:
            if key not in checkpoints_df:
                raise click.ClickException('{} not in: {}'.format(key, list(checkpoints_df)))
        ascending = not desc
        checkpoints_df = checkpoints_df.sort_values(by=sort, ascending=ascending)
    if limit:
        checkpoints_df = checkpoints_df[:limit]
    print_format_output(checkpoints_df)
    if output:
        file_extension = os.path.splitext(output)[1].lower()
        if file_extension in ('.p', '.pkl', '.pickle'):
            checkpoints_df.to_pickle(output)
        elif file_extension == '.csv':
            checkpoints_df.to_csv(output, index=False)
        else:
            raise click.ClickException('Unsupported filetype: {}'.format(output))
        click.secho('Output saved at {}'.format(output), fg='green')