import re
from wasabi import Printer
from ...tokens import Doc, Span, Token
from ...training import biluo_tags_to_spans, iob_to_biluo
from ...vocab import Vocab
from .conll_ner_to_docs import n_sents_info
def conllu_sentence_to_doc(vocab, lines, ner_tag_pattern, merge_subtokens=False, append_morphology=False, ner_map=None, set_ents=False):
    """Create an Example from the lines for one CoNLL-U sentence, merging
    subtokens and appending morphology to tags if required.

    lines (str): The non-comment lines for a CoNLL-U sentence
    ner_tag_pattern (str): The regex pattern for matching NER in MISC col
    RETURNS (Example): An example containing the annotation
    """
    if not Token.has_extension('merged_orth'):
        Token.set_extension('merged_orth', default='')
    if not Token.has_extension('merged_lemma'):
        Token.set_extension('merged_lemma', default='')
    if not Token.has_extension('merged_morph'):
        Token.set_extension('merged_morph', default='')
    if not Token.has_extension('merged_spaceafter'):
        Token.set_extension('merged_spaceafter', default='')
    words, spaces, tags, poses, morphs, lemmas = ([], [], [], [], [], [])
    heads, deps = ([], [])
    subtok_word = ''
    in_subtok = False
    for i in range(len(lines)):
        line = lines[i]
        parts = line.split('\t')
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if '.' in id_:
            continue
        if '-' in id_:
            in_subtok = True
        if '-' in id_:
            in_subtok = True
            subtok_word = word
            subtok_start, subtok_end = id_.split('-')
            subtok_spaceafter = 'SpaceAfter=No' not in misc
            continue
        if merge_subtokens and in_subtok:
            words.append(subtok_word)
        else:
            words.append(word)
        if in_subtok:
            if id_ == subtok_end:
                spaces.append(subtok_spaceafter)
            else:
                spaces.append(False)
        elif 'SpaceAfter=No' in misc:
            spaces.append(False)
        else:
            spaces.append(True)
        if in_subtok and id_ == subtok_end:
            subtok_word = ''
            in_subtok = False
        id_ = int(id_) - 1
        head = int(head) - 1 if head not in ('0', '_') else id_
        tag = pos if tag == '_' else tag
        pos = pos if pos != '_' else ''
        morph = morph if morph != '_' else ''
        dep = 'ROOT' if dep == 'root' else dep
        lemmas.append(lemma)
        poses.append(pos)
        tags.append(tag)
        morphs.append(morph)
        heads.append(head)
        deps.append(dep)
    doc = Doc(vocab, words=words, spaces=spaces, tags=tags, pos=poses, deps=deps, lemmas=lemmas, morphs=morphs, heads=heads)
    for i in range(len(doc)):
        doc[i]._.merged_orth = words[i]
        doc[i]._.merged_morph = morphs[i]
        doc[i]._.merged_lemma = lemmas[i]
        doc[i]._.merged_spaceafter = spaces[i]
    ents = None
    if set_ents:
        ents = get_entities(lines, ner_tag_pattern, ner_map)
        doc.ents = biluo_tags_to_spans(doc, ents)
    if merge_subtokens:
        doc = merge_conllu_subtokens(lines, doc)
    words, spaces, tags, morphs, lemmas, poses = ([], [], [], [], [], [])
    heads, deps = ([], [])
    for i, t in enumerate(doc):
        words.append(t._.merged_orth)
        lemmas.append(t._.merged_lemma)
        spaces.append(t._.merged_spaceafter)
        morphs.append(t._.merged_morph)
        if append_morphology and t._.merged_morph:
            tags.append(t.tag_ + '__' + t._.merged_morph)
        else:
            tags.append(t.tag_)
        poses.append(t.pos_)
        heads.append(t.head.i)
        deps.append(t.dep_)
    doc_x = Doc(vocab, words=words, spaces=spaces, tags=tags, morphs=morphs, lemmas=lemmas, pos=poses, deps=deps, heads=heads)
    if set_ents:
        doc_x.ents = [Span(doc_x, ent.start, ent.end, label=ent.label) for ent in doc.ents]
    return doc_x