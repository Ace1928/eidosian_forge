import math
from string import ascii_letters as LETTERS
from rdkit.sping import pagesizes
from rdkit.sping.pid import *
def redfunc(x):
    return 1.0 / (1.0 + math.exp(10 * (x - 0.5)))