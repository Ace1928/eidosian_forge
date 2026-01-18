import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import wandb
from wandb.util import get_module
def chunkify(input_list, chunk_size) -> List:
    chunk_size = max(1, chunk_size)
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]