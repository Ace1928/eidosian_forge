import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def add_missing_dependencies(self, node, depgraph):
    rel = node['rel'].lower()
    if rel == 'main':
        headnode = depgraph.nodes[node['head']]
        subj = self.lookup_unique('subj', headnode, depgraph)
        relation = subj['rel']
        node['deps'].setdefault(relation, [])
        node['deps'][relation].append(subj['address'])