from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
def comma_separated_string_args(func):
    """
    A decorator that allows a function to be called with
    a single string of comma-separated values which become
    individual function arguments.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        _args = list()
        for arg in args:
            if isinstance(arg, str):
                _args.append({part.strip() for part in arg.split(',')})
            elif isinstance(arg, list):
                _args.append(set(arg))
            else:
                _args.append(arg)
        for name, value in kwargs.items():
            if isinstance(value, str):
                kwargs[name] = {part.strip() for part in value.split(',')}
        return func(*_args, **kwargs)
    return wrapper