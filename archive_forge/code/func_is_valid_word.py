import json
import re
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from itertools import islice
def is_valid_word(self, word):
    """
        Marks whether a string may be included in the unigram list.

        Used to filter punctuation and special tokens.
        """
    return not word.startswith('__') and word != '\n' and (not re.match('[^\\w]', word))