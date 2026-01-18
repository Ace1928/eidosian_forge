import functools
import re
import nltk.tree
def node_label_use_pred(n, m=None, l=None):
    if l is None or node_label not in l:
        raise TgrepException(f'node_label ={node_label} not bound in pattern')
    node = l[node_label]
    return n is node