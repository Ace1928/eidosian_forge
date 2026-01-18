from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def rels_from_rel(R, G):
    """
    Returns the relations in the character variety coming from a relation
    in the group presentation. The input is:

    R - a word object, the relation in the group
    G - a list of words, the set of generators of the group
    """
    relations = [tr(R * g) - tr(g) for g in G]
    relations = relations + [tr(R) - tr(Word(''))]
    return [r for r in relations if r != 0]