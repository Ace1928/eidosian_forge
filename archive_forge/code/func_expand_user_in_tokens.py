import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def expand_user_in_tokens(tokens: List[str]) -> None:
    """
    Call expand_user() on all tokens in a list of strings
    :param tokens: tokens to expand
    """
    for index, _ in enumerate(tokens):
        tokens[index] = expand_user(tokens[index])