import os
import re
import shelve
import sys
import nltk.data
def label_indivs(valuation, lexicon=False):
    """
    Assign individual constants to the individuals in the domain of a ``Valuation``.

    Given a valuation with an entry of the form ``{'rel': {'a': True}}``,
    add a new entry ``{'a': 'a'}``.

    :type valuation: Valuation
    :rtype: Valuation
    """
    domain = valuation.domain
    pairs = [(e, e) for e in domain]
    if lexicon:
        lex = make_lex(domain)
        with open('chat_pnames.cfg', 'w') as outfile:
            outfile.writelines(lex)
    valuation.update(pairs)
    return valuation