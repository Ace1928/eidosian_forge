import argparse
import logging
import sys
from typing import Any, Container, Iterable, List, Optional
import pdfminer.high_level
from pdfminer.layout import LAParams
from pdfminer.utils import AnyIO
def float_or_disabled(x: str) -> Optional[float]:
    if x.lower().strip() == 'disabled':
        return None
    try:
        return float(x)
    except ValueError:
        raise argparse.ArgumentTypeError('invalid float value: {}'.format(x))