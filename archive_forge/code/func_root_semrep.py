import codecs
from nltk.sem import evaluate
def root_semrep(syntree, semkey='SEM'):
    """
    Find the semantic representation at the root of a tree.

    :param syntree: a parse ``Tree``
    :param semkey: the feature label to use for the root semantics in the tree
    :return: the semantic representation at the root of a ``Tree``
    :rtype: sem.Expression
    """
    from nltk.grammar import FeatStructNonterminal
    node = syntree.label()
    assert isinstance(node, FeatStructNonterminal)
    try:
        return node[semkey]
    except KeyError:
        print(node, end=' ')
        print('has no specification for the feature %s' % semkey)
    raise