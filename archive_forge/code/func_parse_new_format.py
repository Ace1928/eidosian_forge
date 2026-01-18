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
def parse_new_format(line):
    num, ((text_indexes, fname_idx, offset), start, end), tag = line
    text = ''.join((text_table[x] for x in text_indexes))
    fname = text_table[fname_idx]
    return (num, ((text, fname, offset), start, end), tag)