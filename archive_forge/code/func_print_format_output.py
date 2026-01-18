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
def print_format_output(dataframe):
    """Prints output of given dataframe to fit into terminal.

    Returns:
        table: Final outputted dataframe.
        dropped_cols: Columns dropped due to terminal size.
        empty_cols: Empty columns (dropped on default).
    """
    print_df = pd.DataFrame()
    dropped_cols = []
    empty_cols = []
    for i, col in enumerate(dataframe):
        if dataframe[col].isnull().all():
            empty_cols += [col]
            continue
        print_df[col] = dataframe[col]
        test_table = tabulate(print_df, headers='keys', tablefmt='psql')
        if str(test_table).index('\n') > TERM_WIDTH:
            print_df.drop(col, axis=1, inplace=True)
            dropped_cols += list(dataframe.columns)[i:]
            break
    table = tabulate(print_df, headers='keys', tablefmt='psql', showindex='never')
    print(table)
    if dropped_cols:
        click.secho('Dropped columns: {}'.format(dropped_cols), fg='yellow')
        click.secho('Please increase your terminal size to view remaining columns.')
    if empty_cols:
        click.secho('Empty columns: {}'.format(empty_cols), fg='yellow')
    return (table, dropped_cols, empty_cols)