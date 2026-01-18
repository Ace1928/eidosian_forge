import time
import numpy as np
import pandas as pd
import pytest
from tqdm.contrib.concurrent import process_map
import panel as pn
from panel.widgets import Tqdm
def run_df(*events):
    df.progress_apply(lambda x: x ** 2)