import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def sentences_from_text_legacy(self, text: str) -> Iterator[str]:
    """
        Given a text, generates the sentences in that text. Annotates all
        tokens, rather than just those with possible sentence breaks. Should
        produce the same results as ``sentences_from_text``.
        """
    tokens = self._annotate_tokens(self._tokenize_words(text))
    return self._build_sentence_list(text, tokens)