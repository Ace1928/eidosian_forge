import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def right_arc(self, conf, relation):
    """
        Note that the algorithm for right-arc is DIFFERENT for arc-standard and arc-eager

        :param configuration: is the current configuration
        :return: A new configuration or -1 if the pre-condition is not satisfied
        """
    if len(conf.buffer) <= 0 or len(conf.stack) <= 0:
        return -1
    if self._algo == TransitionParser.ARC_STANDARD:
        idx_wi = conf.stack.pop()
        idx_wj = conf.buffer[0]
        conf.buffer[0] = idx_wi
        conf.arcs.append((idx_wi, relation, idx_wj))
    else:
        idx_wi = conf.stack[len(conf.stack) - 1]
        idx_wj = conf.buffer.pop(0)
        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))