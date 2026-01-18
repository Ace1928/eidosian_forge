import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def calcLevelSizes(self, levelSizes, level):
    levelSizes[level] += 1
    level += 1
    if level == len(levelSizes):
        return
    for child in self.children:
        child.calcLevelSizes(levelSizes, level)