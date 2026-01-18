import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def rule_based_demo():
    from nltk.grammar import DependencyGrammar
    grammar = DependencyGrammar.fromstring("\n    'taught' -> 'play' | 'man'\n    'man' -> 'the' | 'in'\n    'in' -> 'corner'\n    'corner' -> 'the'\n    'play' -> 'golf' | 'dachshund' | 'to'\n    'dachshund' -> 'his'\n    ")
    print(grammar)
    ndp = NonprojectiveDependencyParser(grammar)
    graphs = ndp.parse(['the', 'man', 'in', 'the', 'corner', 'taught', 'his', 'dachshund', 'to', 'play', 'golf'])
    print('Graphs:')
    for graph in graphs:
        print(graph)