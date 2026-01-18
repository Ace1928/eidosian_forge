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
def rWriteLeaves(self, itemsPerSlot, lNodeSize, tree, curLevel, leafLevel, output):
    formatter_leaf = self.formatter_leaf
    if curLevel == leafLevel:
        isLeaf = True
        data = self.formatter_node.pack(isLeaf, len(tree.children))
        output.write(data)
        for child in tree.children:
            data = formatter_leaf.pack(child.startChromId, child.startBase, child.endChromId, child.endBase, child.startFileOffset, child.endFileOffset - child.startFileOffset)
            output.write(data)
        output.write(bytes((itemsPerSlot - len(tree.children)) * self.formatter_nonleaf.size))
    else:
        for child in tree.children:
            self.rWriteLeaves(itemsPerSlot, lNodeSize, child, curLevel + 1, leafLevel, output)