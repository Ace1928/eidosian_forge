import collections
import copy
import itertools
import random
import re
import warnings
@classmethod
def randomized(cls, taxa, branch_length=1.0, branch_stdev=None):
    """Create a randomized bifurcating tree given a list of taxa.

        :param taxa: Either an integer specifying the number of taxa to create
            (automatically named taxon#), or an iterable of taxon names, as
            strings.

        :returns: a tree of the same type as this class.

        """
    if isinstance(taxa, int):
        taxa = [f'taxon{i + 1}' for i in range(taxa)]
    elif hasattr(taxa, '__iter__'):
        taxa = list(taxa)
    else:
        raise TypeError('taxa argument must be integer (# taxa) or iterable of taxon names.')
    rtree = cls()
    terminals = [rtree.root]
    while len(terminals) < len(taxa):
        newsplit = random.choice(terminals)
        newsplit.split(branch_length=branch_length)
        newterms = newsplit.clades
        if branch_stdev:
            for nt in newterms:
                nt.branch_length = max(0, random.gauss(branch_length, branch_stdev))
        terminals.remove(newsplit)
        terminals.extend(newterms)
    random.shuffle(taxa)
    for node, name in zip(terminals, taxa):
        node.name = name
    return rtree