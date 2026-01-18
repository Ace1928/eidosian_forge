from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
def apply_in_place(self, im):
    im.load()
    if im.mode != self.output_mode:
        msg = 'mode mismatch'
        raise ValueError(msg)
    self.transform.apply(im.im.id, im.im.id)
    im.info['icc_profile'] = self.output_profile.tobytes()
    return im