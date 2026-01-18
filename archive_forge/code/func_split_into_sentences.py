import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def split_into_sentences(text, ensure_compatibility, language='english'):
    """
        Split text into sentences, using specified language. Use PunktSentenceTokenizer

        Args:
          text: The string text to tokenize
          ensure_compatibility: Split sentences by '
' instead of NLTK sentence tokenizer model
          language: Language of the text

        Returns:
          List of tokens of text
        """
    if ensure_compatibility:
        return text.split('\n')
    else:
        return nltk.sent_tokenize(text, language)