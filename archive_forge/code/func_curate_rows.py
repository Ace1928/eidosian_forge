import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def curate_rows(rows: Set[int], GRID) -> List:
    rows = list(rows)
    rows.sort()
    nrows = [rows[0]]
    for h in rows[1:]:
        if h >= nrows[-1] + GRID:
            nrows.append(h)
    return nrows