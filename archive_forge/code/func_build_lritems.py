import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def build_lritems(self):
    for p in self.Productions:
        lastlri = p
        i = 0
        lr_items = []
        while True:
            if i > len(p):
                lri = None
            else:
                lri = LRItem(p, i)
                try:
                    lri.lr_after = self.Prodnames[lri.prod[i + 1]]
                except (IndexError, KeyError):
                    lri.lr_after = []
                try:
                    lri.lr_before = lri.prod[i - 1]
                except IndexError:
                    lri.lr_before = None
            lastlri.lr_next = lri
            if not lri:
                break
            lr_items.append(lri)
            lastlri = lri
            i += 1
        p.lr_items = lr_items