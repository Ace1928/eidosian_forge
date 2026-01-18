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
def print_mean_with_ci(sample: numpy.ndarray):
    mean = numpy.mean(sample)
    bootstrap_means = bootstrap(sample)
    bootstrap_means.sort()
    low = bootstrap_means[int(len(bootstrap_means) * 0.025)]
    high = bootstrap_means[int(len(bootstrap_means) * 0.975)]
    print(f'Mean: {mean:.1f} words/s (95% CI: {low - mean:.1f} +{high - mean:.1f})')