import sys
import os
import io
import pathlib
import re
import argparse
import zipfile
import json
import pickle
import pprint
import urllib.parse
from typing import (
import torch.utils.show_pickle
def get_pickle(name):
    assert path_prefix is not None
    with zf.open(path_prefix + f'/{name}.pkl') as handle:
        raw = torch.utils.show_pickle.DumpUnpickler(handle, catch_invalid_utf8=True).load()
        return hierarchical_pickle(raw)