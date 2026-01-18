import json
import logging
import re
import uuid
import warnings
from base64 import b64encode
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import UNPKG_DT_BUNDLE_CSS, UNPKG_DT_BUNDLE_URL
from .version import __version__ as itables_version
from IPython.display import HTML, display
import itables.options as opt
from .datatables_format import datatables_rows
from .downsample import downsample
from .utils import read_package_file
def safe_reset_index(df):
    try:
        return df.reset_index()
    except ValueError:
        index_levels = [pd.Series(df.index.get_level_values(i), name=name or ('index{}'.format(i) if isinstance(df.index, pd.MultiIndex) else 'index')) for i, name in enumerate(df.index.names)]
        return pd.concat(index_levels + [df.reset_index(drop=True)], axis=1)