import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@property
def first_case(self):
    if self.first_lower:
        return 'lower'
    if self.first_upper:
        return 'upper'
    return 'none'