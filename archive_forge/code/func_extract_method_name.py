import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def extract_method_name(line: str) -> str:
    """Extract method name from decorator in the form of "@functional_datapipe({method_name})"."""
    if '("' in line:
        start_token, end_token = ('("', '")')
    elif "('" in line:
        start_token, end_token = ("('", "')")
    else:
        raise RuntimeError(f'Unable to find appropriate method name within line:\n{line}')
    start, end = (line.find(start_token) + len(start_token), line.find(end_token))
    return line[start:end]