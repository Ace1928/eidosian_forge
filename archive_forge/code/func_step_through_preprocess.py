from __future__ import with_statement
import logging
import os
import random
import re
import sys
from gensim import interfaces, utils
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import (
from gensim.utils import deaccent, simple_tokenize
from smart_open import open
def step_through_preprocess(self, text):
    """Apply preprocessor one by one and generate result.

        Warnings
        --------
        This is useful for debugging issues with the corpus preprocessing pipeline.

        Parameters
        ----------
        text : str
            Document text read from plain-text file.

        Yields
        ------
        (callable, object)
            Pre-processor, output from pre-processor (based on `text`)

        """
    for character_filter in self.character_filters:
        text = character_filter(text)
        yield (character_filter, text)
    tokens = self.tokenizer(text)
    yield (self.tokenizer, tokens)
    for token_filter in self.token_filters:
        yield (token_filter, token_filter(tokens))