import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import filesystem, runid
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
def is_power2(num: int) -> bool:
    return num != 0 and num & num - 1 == 0