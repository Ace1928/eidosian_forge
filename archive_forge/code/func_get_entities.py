import re
from wasabi import Printer
from ...tokens import Doc, Span, Token
from ...training import biluo_tags_to_spans, iob_to_biluo
from ...vocab import Vocab
from .conll_ner_to_docs import n_sents_info
def get_entities(lines, tag_pattern, ner_map=None):
    """Find entities in the MISC column according to the pattern and map to
    final entity type with `ner_map` if mapping present. Entity tag is 'O' if
    the pattern is not matched.

    lines (str): CONLL-U lines for one sentences
    tag_pattern (str): Regex pattern for entity tag
    ner_map (dict): Map old NER tag names to new ones, '' maps to O.
    RETURNS (list): List of BILUO entity tags
    """
    miscs = []
    for line in lines:
        parts = line.split('\t')
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if '-' in id_ or '.' in id_:
            continue
        miscs.append(misc)
    iob = []
    for misc in miscs:
        iob_tag = 'O'
        for misc_part in misc.split('|'):
            tag_match = re.match(tag_pattern, misc_part)
            if tag_match:
                prefix = tag_match.group(2)
                suffix = tag_match.group(3)
                if prefix and suffix:
                    iob_tag = prefix + '-' + suffix
                    if ner_map:
                        suffix = ner_map.get(suffix, suffix)
                        if suffix == '':
                            iob_tag = 'O'
                        else:
                            iob_tag = prefix + '-' + suffix
                break
        iob.append(iob_tag)
    return iob_to_biluo(iob)