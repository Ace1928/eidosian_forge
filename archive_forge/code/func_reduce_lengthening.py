import html
import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple
import regex
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    pattern = regex.compile('(.)\\1{2,}')
    return pattern.sub('\\1\\1\\1', text)