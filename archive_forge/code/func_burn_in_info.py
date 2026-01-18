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
def burn_in_info(skeleton, info):
    """Burn model info into the HTML skeleton.

    The result will render the hard-coded model info and
    have no external network dependencies for code or data.
    """
    return skeleton.replace('BURNED_IN_MODEL_INFO = null', 'BURNED_IN_MODEL_INFO = ' + json.dumps(info, sort_keys=True).replace('/', '\\/'))