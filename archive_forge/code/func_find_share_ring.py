from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def find_share_ring(self, tree_map, parent_map, r):
    """
        get a ring structure that tends to share nodes with the tree
        return a list starting from r
        """
    nset = set(tree_map[r])
    cset = nset - set([parent_map[r]])
    if len(cset) == 0:
        return [r]
    rlst = [r]
    cnt = 0
    for v in cset:
        vlst = self.find_share_ring(tree_map, parent_map, v)
        cnt += 1
        if cnt == len(cset):
            vlst.reverse()
        rlst += vlst
    return rlst