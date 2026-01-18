import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging
@property
def question_token_id(self):
    """
        `Optional[int]`: Id of the question token in the vocabulary, used to condition the answer on a question
        representation.
        """
    return self.convert_tokens_to_ids(self.question_token)