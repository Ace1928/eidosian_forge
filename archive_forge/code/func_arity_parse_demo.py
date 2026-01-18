from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
def arity_parse_demo():
    """
    A demonstration showing the creation of a ``DependencyGrammar``
    in which a specific number of modifiers is listed for a given
    head.  This can further constrain the number of possible parses
    created by a ``ProjectiveDependencyParser``.
    """
    print()
    print('A grammar with no arity constraints. Each DependencyProduction')
    print('specifies a relationship between one head word and only one')
    print('modifier word.')
    grammar = DependencyGrammar.fromstring("\n    'fell' -> 'price' | 'stock'\n    'price' -> 'of' | 'the'\n    'of' -> 'stock'\n    'stock' -> 'the'\n    ")
    print(grammar)
    print()
    print("For the sentence 'The price of the stock fell', this grammar")
    print('will produce the following three parses:')
    pdp = ProjectiveDependencyParser(grammar)
    trees = pdp.parse(['the', 'price', 'of', 'the', 'stock', 'fell'])
    for tree in trees:
        print(tree)
    print()
    print('By contrast, the following grammar contains a ')
    print('DependencyProduction that specifies a relationship')
    print("between a single head word, 'price', and two modifier")
    print("words, 'of' and 'the'.")
    grammar = DependencyGrammar.fromstring("\n    'fell' -> 'price' | 'stock'\n    'price' -> 'of' 'the'\n    'of' -> 'stock'\n    'stock' -> 'the'\n    ")
    print(grammar)
    print()
    print('This constrains the number of possible parses to just one:')
    pdp = ProjectiveDependencyParser(grammar)
    trees = pdp.parse(['the', 'price', 'of', 'the', 'stock', 'fell'])
    for tree in trees:
        print(tree)