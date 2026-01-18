import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def hall_demo():
    npp = ProbabilisticNonprojectiveParser()
    npp.train([], DemoScorer())
    for parse_graph in npp.parse(['v1', 'v2', 'v3'], [None, None, None]):
        print(parse_graph)