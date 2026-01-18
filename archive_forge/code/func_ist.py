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
def ist(s):
    if s not in interned_strings:
        interned_strings[s] = len(interned_strings)
    return interned_strings[s]