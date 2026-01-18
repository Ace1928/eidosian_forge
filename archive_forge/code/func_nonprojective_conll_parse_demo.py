import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def nonprojective_conll_parse_demo():
    from nltk.parse.dependencygraph import conll_data2
    graphs = [DependencyGraph(entry) for entry in conll_data2.split('\n\n') if entry]
    npp = ProbabilisticNonprojectiveParser()
    npp.train(graphs, NaiveBayesDependencyScorer())
    for parse_graph in npp.parse(['Cathy', 'zag', 'hen', 'zwaaien', '.'], ['N', 'V', 'Pron', 'Adj', 'N', 'Punc']):
        print(parse_graph)