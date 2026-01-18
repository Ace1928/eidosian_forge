from glob import glob
from os.path import isdir
from sys import argv
from typing import List
import argparse
from charset_normalizer import detect as tbt_detect
from chardet import detect as chardet_detect
from charset_normalizer.utils import iana_name
def calc_equivalence(content: bytes, cp_a: str, cp_b: str):
    try:
        str_a = content.decode(cp_a)
        str_b = content.decode(cp_b)
    except UnicodeDecodeError:
        return 0.0
    character_count = len(str_a)
    diff_character_count = sum((chr_a != chr_b for chr_a, chr_b in zip(str_a, str_b)))
    return 1.0 - diff_character_count / character_count