import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from typing_extensions import override
from pytorch_lightning.profilers.profiler import Profiler
def log_row_extended(action: str, mean: str, num_calls: str, total: str, per: str) -> str:
    row = f'{sep}|  {action:<{max_key}s}\t|  {mean:<15}\t|'
    row += f'  {num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|'
    return row