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
def init_dictionary(self, dictionary):
    """Initialize/update dictionary.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.

        Notes
        -----
        If self.input is None - make nothing.

        """
    self.dictionary = dictionary if dictionary is not None else Dictionary()
    if self.input is not None:
        if dictionary is None:
            logger.info('Initializing dictionary')
            metadata_setting = self.metadata
            self.metadata = False
            self.dictionary.add_documents(self.get_texts())
            self.metadata = metadata_setting
        else:
            logger.info('Input stream provided but dictionary already initialized')
    else:
        logger.warning('No input document stream provided; assuming dictionary will be initialized some other way.')