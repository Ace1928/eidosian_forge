import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
def fp16_apex_available() -> bool:
    try:
        import apex.fp16_utils
        return True
    except ImportError:
        error_once('You set --fp16 true with --fp16-impl apex, but fp16 with apex is unavailable. To use apex fp16, please install APEX from https://github.com/NVIDIA/apex.')
        return False