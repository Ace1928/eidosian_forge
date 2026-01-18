from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os
def scan_over_(self, valid):
    p = self.pos_
    while p < self.text_length_ and self.text_[p] in valid:
        p += 1
    self.pos_ = p