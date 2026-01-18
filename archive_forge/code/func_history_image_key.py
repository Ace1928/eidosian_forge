import io
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.util
from wandb.sdk.lib import telemetry
from wandb.viz import custom_chart
def history_image_key(key: str, namespace: str='') -> str:
    """Convert invalid filesystem characters to _ for use in History keys.

    Unfortunately this means currently certain image keys will collide silently. We
    implement this mapping up here in the TensorFlow stuff rather than in the History
    stuff so that we don't have to store a mapping anywhere from the original keys to
    the safe ones.
    """
    return namespaced_tag(re.sub('[/\\\\]', '_', key), namespace)