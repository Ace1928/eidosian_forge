from typing import List, Tuple
import pytest
from nltk.tokenize import (
def load_stanford_segmenter():
    try:
        seg = StanfordSegmenter()
        seg.default_config('ar')
        seg.default_config('zh')
        return True
    except LookupError:
        return False