import math
import re
from nltk.tokenize.api import TokenizerI
def blk_frq(tok, block):
    ts_occs = filter(lambda o: o[0] in block, token_table[tok].ts_occurences)
    freq = sum((tsocc[1] for tsocc in ts_occs))
    return freq