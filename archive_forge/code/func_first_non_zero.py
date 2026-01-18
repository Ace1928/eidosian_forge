import os
from random import choice
from itertools import combinations
import snappy
from plink import LinkManager
from .twister_core import build_bundle, build_splitting, twister_version
def first_non_zero(L):
    return min((i for i in range(len(L)) if L[i]))