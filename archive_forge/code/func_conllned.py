import html
import re
from collections import defaultdict
def conllned(trace=1):
    """
    Find the copula+'van' relation ('of') in the Dutch tagged training corpus
    from CoNLL 2002.
    """
    from nltk.corpus import conll2002
    vnv = "\n    (\n    is/V|    # 3rd sing present and\n    was/V|   # past forms of the verb zijn ('be')\n    werd/V|  # and also present\n    wordt/V  # past of worden ('become)\n    )\n    .*       # followed by anything\n    van/Prep # followed by van ('of')\n    "
    VAN = re.compile(vnv, re.VERBOSE)
    print()
    print('Dutch CoNLL2002: van(PER, ORG) -- raw rtuples with context:')
    print('=' * 45)
    for doc in conll2002.chunked_sents('ned.train'):
        lcon = rcon = False
        if trace:
            lcon = rcon = True
        for rel in extract_rels('PER', 'ORG', doc, corpus='conll2002', pattern=VAN, window=10):
            print(rtuple(rel, lcon=lcon, rcon=rcon))