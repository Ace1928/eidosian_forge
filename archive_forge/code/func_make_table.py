import time
from warnings import simplefilter
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
import wandb
def make_table(cluster_ranges, clfs, times):
    columns = ['cluster_ranges', 'errors', 'clustering_time']
    data = list(zip(cluster_ranges, clfs, times))
    table = wandb.Table(columns=columns, data=data)
    return table