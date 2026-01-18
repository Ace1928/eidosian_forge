import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def period_context_re(self):
    """Compiles and returns a regular expression to find contexts
        including possible sentence boundaries."""
    try:
        return self._re_period_context
    except:
        self._re_period_context = re.compile(self._period_context_fmt % {'NonWord': self._re_non_word_chars, 'SentEndChars': self._re_sent_end_chars}, re.UNICODE | re.VERBOSE)
        return self._re_period_context