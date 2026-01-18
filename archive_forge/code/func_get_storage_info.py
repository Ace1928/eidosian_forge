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
def get_storage_info(storage):
    assert isinstance(storage, torch.utils.show_pickle.FakeObject)
    assert storage.module == 'pers'
    assert storage.name == 'obj'
    assert storage.state is None
    assert isinstance(storage.args, tuple)
    assert len(storage.args) == 1
    sa = storage.args[0]
    assert isinstance(sa, tuple)
    assert len(sa) == 5
    assert sa[0] == 'storage'
    assert isinstance(sa[1], torch.utils.show_pickle.FakeClass)
    assert sa[1].module == 'torch'
    assert sa[1].name.endswith('Storage')
    storage_info = [sa[1].name.replace('Storage', '')] + list(sa[2:])
    return storage_info