import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client
import xgboost as xgb
from xgboost.testing.updater import get_basescore
def check_init_estimation(tree_method: str, client: Client) -> None:
    """Test init estimation."""
    check_init_estimation_reg(tree_method, client)
    check_init_estimation_clf(tree_method, client)