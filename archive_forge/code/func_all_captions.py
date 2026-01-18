import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union, cast
from urllib import parse
import wandb
from wandb import util
from wandb.sdk.lib import hashutil, runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
from .helper_types.bounding_boxes_2d import BoundingBoxes2D
from .helper_types.classes import Classes
from .helper_types.image_mask import ImageMask
@classmethod
def all_captions(cls: Type['Image'], images: Sequence['Media']) -> Union[bool, Sequence[Optional[str]]]:
    return cls.captions(images)