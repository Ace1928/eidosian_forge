import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, dviread
@classmethod
def get_custom_preamble(cls):
    """Return a string containing user additions to the tex preamble."""
    return mpl.rcParams['text.latex.preamble']