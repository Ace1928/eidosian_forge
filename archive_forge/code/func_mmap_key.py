import json
import mmap
import os
import struct
from typing import List
def mmap_key(metric_name: str, name: str, labelnames: List[str], labelvalues: List[str], help_text: str) -> str:
    """Format a key for use in the mmap file."""
    labels = dict(zip(labelnames, labelvalues))
    return json.dumps([metric_name, name, labels, help_text], sort_keys=True)