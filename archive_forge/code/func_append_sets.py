from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def append_sets(self, exclude=(), delete=(), mrbayes=False, include_codons=True, codons_only=False):
    """Return a sets block."""
    if not self.charsets and (not self.taxsets) and (not self.charpartitions):
        return ''
    if codons_only:
        setsb = ['\nbegin codons']
    else:
        setsb = ['\nbegin sets']
    offset = 0
    offlist = []
    for c in range(self.nchar):
        if c in exclude:
            offset += 1
            offlist.append(-1)
        else:
            offlist.append(c - offset)
    if not codons_only:
        for n, ns in self.charsets.items():
            cset = [offlist[c] for c in ns if c not in exclude]
            if cset:
                setsb.append(f'charset {safename(n)} = {_compact4nexus(cset)}')
        for n, s in self.taxsets.items():
            tset = [safename(t, mrbayes=mrbayes) for t in s if t not in delete]
            if tset:
                setsb.append(f'taxset {safename(n)} = {' '.join(tset)}')
    for n, p in self.charpartitions.items():
        if not include_codons and n == CODONPOSITIONS:
            continue
        elif codons_only and n != CODONPOSITIONS:
            continue
        names = _sort_keys_by_values(p)
        newpartition = {}
        for sn in names:
            nsp = [offlist[c] for c in p[sn] if c not in exclude]
            if nsp:
                newpartition[sn] = nsp
        if newpartition:
            if include_codons and n == CODONPOSITIONS:
                command = 'codonposset'
            else:
                command = 'charpartition'
            setsb.append('%s %s = %s' % (command, safename(n), ', '.join((f'{sn}: {_compact4nexus(newpartition[sn])}' for sn in names if sn in newpartition))))
    for n, p in self.taxpartitions.items():
        names = _sort_keys_by_values(p)
        newpartition = {}
        for sn in names:
            nsp = [t for t in p[sn] if t not in delete]
            if nsp:
                newpartition[sn] = nsp
        if newpartition:
            setsb.append('taxpartition %s = %s' % (safename(n), ', '.join(('%s: %s' % (safename(sn), ' '.join((safename(x) for x in newpartition[sn]))) for sn in names if sn in newpartition))))
    setsb.append('end;\n')
    if len(setsb) == 2:
        return ''
    else:
        return ';\n'.join(setsb)