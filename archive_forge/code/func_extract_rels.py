import html
import re
from collections import defaultdict
def extract_rels(subjclass, objclass, doc, corpus='ace', pattern=None, window=10):
    """
    Filter the output of ``semi_rel2reldict`` according to specified NE classes and a filler pattern.

    The parameters ``subjclass`` and ``objclass`` can be used to restrict the
    Named Entities to particular types (any of 'LOCATION', 'ORGANIZATION',
    'PERSON', 'DURATION', 'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE').

    :param subjclass: the class of the subject Named Entity.
    :type subjclass: str
    :param objclass: the class of the object Named Entity.
    :type objclass: str
    :param doc: input document
    :type doc: ieer document or a list of chunk trees
    :param corpus: name of the corpus to take as input; possible values are
        'ieer' and 'conll2002'
    :type corpus: str
    :param pattern: a regular expression for filtering the fillers of
        retrieved triples.
    :type pattern: SRE_Pattern
    :param window: filters out fillers which exceed this threshold
    :type window: int
    :return: see ``mk_reldicts``
    :rtype: list(defaultdict)
    """
    if subjclass and subjclass not in NE_CLASSES[corpus]:
        if _expand(subjclass) in NE_CLASSES[corpus]:
            subjclass = _expand(subjclass)
        else:
            raise ValueError('your value for the subject type has not been recognized: %s' % subjclass)
    if objclass and objclass not in NE_CLASSES[corpus]:
        if _expand(objclass) in NE_CLASSES[corpus]:
            objclass = _expand(objclass)
        else:
            raise ValueError('your value for the object type has not been recognized: %s' % objclass)
    if corpus == 'ace' or corpus == 'conll2002':
        pairs = tree2semi_rel(doc)
    elif corpus == 'ieer':
        pairs = tree2semi_rel(doc.text) + tree2semi_rel(doc.headline)
    else:
        raise ValueError('corpus type not recognized')
    reldicts = semi_rel2reldict(pairs)
    relfilter = lambda x: x['subjclass'] == subjclass and len(x['filler'].split()) <= window and pattern.match(x['filler']) and (x['objclass'] == objclass)
    return list(filter(relfilter, reldicts))