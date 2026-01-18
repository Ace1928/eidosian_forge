import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def sentences_from_text(self, text: str, realign_boundaries: bool=True) -> List[str]:
    """
        Given a text, generates the sentences in that text by only
        testing candidate sentence breaks. If realign_boundaries is
        True, includes in the sentence closing punctuation that
        follows the period.
        """
    return [text[s:e] for s, e in self.span_tokenize(text, realign_boundaries)]