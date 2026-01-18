import os
from .regexp_tokenizer import RegexpTokenizer
from .simple_tokenizer import SimpleTokenizer
from .corenlp_tokenizer import CoreNLPTokenizer  # noqa: E402
def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators