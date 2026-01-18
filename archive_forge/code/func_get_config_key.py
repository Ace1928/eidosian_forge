from typing import Any, Dict, Optional, Tuple
from wandb.data_types import Table
from wandb.errors import Error
@staticmethod
def get_config_key(key: str) -> Tuple[str, str, str]:
    return ('_wandb', 'visualize', key)