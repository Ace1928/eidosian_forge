import re
from typing import Tuple
from nltk.stem.api import StemmerI
@staticmethod
def replace_to(word: str) -> str:
    word = word.replace('sch', '$')
    word = word.replace('ei', '%')
    word = word.replace('ie', '&')
    word = Cistem.repl_xx.sub('\\1*', word)
    return word