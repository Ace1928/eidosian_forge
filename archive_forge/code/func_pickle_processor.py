from __future__ import absolute_import, division, print_function
import argparse
import itertools as it
import multiprocessing as mp
import os
import sys
from collections import MutableSequence
import numpy as np
from .utils import integer_types
def pickle_processor(processor, outfile, **kwargs):
    """
    Pickle the Processor to a file.

    Parameters
    ----------
    processor : :class:`Processor` instance
        Processor to be pickled.
    outfile : str or file handle
        Output file (handle) where to pickle it.

    """
    processor.dump(outfile)