from __future__ import annotations
import time
import os, os.path, sys, glob, argparse, resource, multiprocessing
import pandas as pd
import dask.dataframe as dd
import numpy as np
import datashader as ds
import feather
import fastparquet as fp
from datashader.utils import export_image
from datashader import transfer_functions as tf
from dask import distributed
def read_csv_pandas(filepath, usecols=None):
    if os.path.isfile(filepath):
        return pd.read_csv(filepath, usecols=usecols)
    filepath_expr = filepath.replace('.csv', '*.csv')
    filepaths = glob.glob(filepath_expr)
    return pd.concat((pd.read_csv(f, usecols=usecols) for f in filepaths))