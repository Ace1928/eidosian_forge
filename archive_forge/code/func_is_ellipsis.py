import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@property
def is_ellipsis(self):
    """True if the token text is that of an ellipsis."""
    return self._RE_ELLIPSIS.match(self.tok)