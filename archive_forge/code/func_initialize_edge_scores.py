import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def initialize_edge_scores(self, graph):
    """
        Assigns a score to every edge in the ``DependencyGraph`` graph.
        These scores are generated via the parser's scorer which
        was assigned during the training process.

        :type graph: DependencyGraph
        :param graph: A dependency graph to assign scores to.
        """
    self.scores = self._scorer.score(graph)