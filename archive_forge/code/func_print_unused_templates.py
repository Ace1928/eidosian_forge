from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def print_unused_templates():
    usedtpls = {int(tid) for tid in tids}
    unused = [(tid, tpl) for tid, tpl in enumerate(Template.ALLTEMPLATES) if tid not in usedtpls]
    print(f'UNUSED TEMPLATES ({len(unused)})')
    for tid, tpl in unused:
        print(f'{tid:03d} {str(tpl):s}')