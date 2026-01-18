import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def mycenter(x):
    return (' %s ' % x).center(75, '-')