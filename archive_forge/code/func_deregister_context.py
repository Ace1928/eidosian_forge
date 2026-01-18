import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def deregister_context(self, context_words):
    """Deregister a list of context words.

    Args:
      context_words: A list of context words to deregister, as a list of str.

    Raises:
      KeyError: if there are word(s) in context_words that do not correspond
        to any registered contexts.
    """
    for context_word in context_words:
        if context_word not in self._comp_dict:
            raise KeyError('Cannot deregister unregistered context word "%s"' % context_word)
    for context_word in context_words:
        del self._comp_dict[context_word]