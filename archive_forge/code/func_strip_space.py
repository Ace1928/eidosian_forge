import contextlib
import sys
import unittest
from io import StringIO
from nltk.corpus import gutenberg
from nltk.text import Text
def strip_space(raw_str):
    return raw_str.replace(' ', '')