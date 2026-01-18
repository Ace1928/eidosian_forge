import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def freq_threshold(self, ortho_thresh=2, type_thresh=2, colloc_thres=2, sentstart_thresh=2):
    """
        Allows memory use to be reduced after much training by removing data
        about rare tokens that are unlikely to have a statistical effect with
        further training. Entries occurring above the given thresholds will be
        retained.
        """
    if ortho_thresh > 1:
        old_oc = self._params.ortho_context
        self._params.clear_ortho_context()
        for tok in self._type_fdist:
            count = self._type_fdist[tok]
            if count >= ortho_thresh:
                self._params.ortho_context[tok] = old_oc[tok]
    self._type_fdist = self._freq_threshold(self._type_fdist, type_thresh)
    self._collocation_fdist = self._freq_threshold(self._collocation_fdist, colloc_thres)
    self._sent_starter_fdist = self._freq_threshold(self._sent_starter_fdist, sentstart_thresh)