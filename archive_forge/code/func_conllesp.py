import html
import re
from collections import defaultdict
def conllesp():
    from nltk.corpus import conll2002
    de = '\n    .*\n    (\n    de/SP|\n    del/SP\n    )\n    '
    DE = re.compile(de, re.VERBOSE)
    print()
    print('Spanish CoNLL2002: de(ORG, LOC) -- just the first 10 clauses:')
    print('=' * 45)
    rels = [rel for doc in conll2002.chunked_sents('esp.train') for rel in extract_rels('ORG', 'LOC', doc, corpus='conll2002', pattern=DE)]
    for r in rels[:10]:
        print(clause(r, relsym='DE'))
    print()