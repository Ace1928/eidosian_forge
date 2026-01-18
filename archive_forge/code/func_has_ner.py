import re
from wasabi import Printer
from ...tokens import Doc, Span, Token
from ...training import biluo_tags_to_spans, iob_to_biluo
from ...vocab import Vocab
from .conll_ner_to_docs import n_sents_info
def has_ner(input_data, ner_tag_pattern):
    """
    Check the MISC column for NER tags.
    """
    for sent in input_data.strip().split('\n\n'):
        lines = sent.strip().split('\n')
        if lines:
            while lines[0].startswith('#'):
                lines.pop(0)
            for line in lines:
                parts = line.split('\t')
                id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
                for misc_part in misc.split('|'):
                    if re.match(ner_tag_pattern, misc_part):
                        return True
    return False