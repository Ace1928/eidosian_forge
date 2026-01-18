import functools
import re
import nltk.tree
def pattern_segment_pred(n, m=None, l=None):
    """This predicate function ignores its node argument."""
    if l is None or node_label not in l:
        raise TgrepException(f'node_label ={node_label} not bound in pattern')
    node = l[node_label]
    return all((pred(node, m, l) for pred in reln_preds))