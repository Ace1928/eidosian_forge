import sys
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import srsly
import tqdm
from wasabi import Printer
from .. import util
from ..errors import Errors
from ..util import registry
def setup_table(*, cols: List[str], widths: List[int], max_width: int=13) -> Tuple[List[str], List[int], List[str]]:
    final_cols = []
    final_widths = []
    for col, width in zip(cols, widths):
        if len(col) > max_width:
            col = col[:max_width - 3] + '...'
        final_cols.append(col.upper())
        final_widths.append(max(len(col), width))
    return (final_cols, final_widths, ['r' for _ in final_widths])