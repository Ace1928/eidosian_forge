import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def lookup_unique(self, rel, node, depgraph):
    """
        Lookup 'key'. There should be exactly one item in the associated relation.
        """
    deps = [depgraph.nodes[dep] for dep in chain.from_iterable(node['deps'].values()) if depgraph.nodes[dep]['rel'].lower() == rel.lower()]
    if len(deps) == 0:
        raise KeyError("'{}' doesn't contain a feature '{}'".format(node['word'], rel))
    elif len(deps) > 1:
        raise KeyError("'{}' should only have one feature '{}'".format(node['word'], rel))
    else:
        return deps[0]