import numbers
import os
from typing import TYPE_CHECKING, Optional, Type, Union
import wandb
from wandb import util
from wandb.sdk.lib import runid
from .._private import MEDIA_TMP
from ..base_types.media import Media
@classmethod
def get_media_subdir(cls: Type['ImageMask']) -> str:
    return os.path.join('media', 'images', cls.type_name())