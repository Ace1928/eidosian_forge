import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def left_arc(self, conf, relation):
    """
        Note that the algorithm for left-arc is quite similar except for precondition for both arc-standard and arc-eager

        :param configuration: is the current configuration
        :return: A new configuration or -1 if the pre-condition is not satisfied
        """
    if len(conf.buffer) <= 0 or len(conf.stack) <= 0:
        return -1
    if conf.buffer[0] == 0:
        return -1
    idx_wi = conf.stack[len(conf.stack) - 1]
    flag = True
    if self._algo == TransitionParser.ARC_EAGER:
        for idx_parent, r, idx_child in conf.arcs:
            if idx_child == idx_wi:
                flag = False
    if flag:
        conf.stack.pop()
        idx_wj = conf.buffer[0]
        conf.arcs.append((idx_wj, relation, idx_wi))
    else:
        return -1