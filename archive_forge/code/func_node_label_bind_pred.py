import functools
import re
import nltk.tree
def node_label_bind_pred(n, m=None, l=None):
    if node_pred(n, m, l):
        if l is None:
            raise TgrepException('cannot bind node_label {}: label_dict is None'.format(node_label))
        l[node_label] = n
        return True
    else:
        return False