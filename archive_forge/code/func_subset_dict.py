import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def subset_dict(original_dict: Dict[str, Any], keys_subset: Sequence[str]) -> Dict[str, Any]:
    """Create a subset of a dictionary using a subset of keys.

    :param original_dict: The original dictionary.
    :param keys_subset: The subset of keys to extract.
    :return: A dictionary containing only the specified keys.
    """
    return {key: original_dict[key] for key in keys_subset if key in original_dict}