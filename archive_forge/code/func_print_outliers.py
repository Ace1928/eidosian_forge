import random
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional
import numpy
import typer
from tqdm import tqdm
from wasabi import msg
from .. import util
from ..language import Language
from ..tokens import Doc
from ..training import Corpus
from ._util import Arg, Opt, benchmark_cli, import_code, setup_gpu
def print_outliers(sample: numpy.ndarray):
    quartiles = Quartiles(sample)
    n_outliers = numpy.sum((sample < quartiles.q1 - 1.5 * quartiles.iqr) | (sample > quartiles.q3 + 1.5 * quartiles.iqr))
    n_extreme_outliers = numpy.sum((sample < quartiles.q1 - 3.0 * quartiles.iqr) | (sample > quartiles.q3 + 3.0 * quartiles.iqr))
    print(f'Outliers: {100 * n_outliers / len(sample):.1f}%, extreme outliers: {100 * n_extreme_outliers / len(sample)}%')