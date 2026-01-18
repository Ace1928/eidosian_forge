import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def get_completions(self, context_word, prefix):
    """Get the tab completions given a context word and a prefix.

    Args:
      context_word: The context word.
      prefix: The prefix of the incomplete word.

    Returns:
      (1) None if no registered context matches the context_word.
          A list of str for the matching completion items. Can be an empty list
          of a matching context exists, but no completion item matches the
          prefix.
      (2) Common prefix of all the words in the first return value. If the
          first return value is None, this return value will be None, too. If
          the first return value is not None, i.e., a list, this return value
          will be a str, which can be an empty str if there is no common
          prefix among the items of the list.
    """
    if context_word not in self._comp_dict:
        return (None, None)
    comp_items = self._comp_dict[context_word]
    comp_items = sorted([item for item in comp_items if item.startswith(prefix)])
    return (comp_items, self._common_prefix(comp_items))