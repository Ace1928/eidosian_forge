from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from collections import defaultdict
import copy
import os
import re
def re_tokenize(self, text):
    """
        This splits along whitespace and punctuation and keeps the newline as a token in
        the returned list.
        """
    return RETOK.findall(text)