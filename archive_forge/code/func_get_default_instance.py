import re
from threading import Lock
from io import TextIOBase
from sqlparse import tokens, keywords
from sqlparse.utils import consume
@classmethod
def get_default_instance(cls):
    """Returns the lexer instance used internally
        by the sqlparse core functions."""
    with cls._lock:
        if cls._default_instance is None:
            cls._default_instance = cls()
            cls._default_instance.default_initialization()
    return cls._default_instance