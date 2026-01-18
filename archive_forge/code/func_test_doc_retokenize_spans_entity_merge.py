import pytest
from spacy.attrs import LEMMA
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_spans_entity_merge(en_tokenizer):
    text = 'Stewart Lee is a stand up comedian who lives in England and loves Joe Pasquale.\n'
    heads = [1, 2, 2, 4, 6, 4, 2, 8, 6, 8, 9, 8, 8, 14, 12, 2, 15]
    deps = ['dep'] * len(heads)
    tags = ['NNP', 'NNP', 'VBZ', 'DT', 'VB', 'RP', 'NN', 'WP', 'VBZ', 'IN', 'NNP', 'CC', 'VBZ', 'NNP', 'NNP', '.', 'SP']
    ents = [('PERSON', 0, 2), ('GPE', 10, 11), ('PERSON', 13, 15)]
    ents = ['O'] * len(heads)
    ents[0] = 'B-PERSON'
    ents[1] = 'I-PERSON'
    ents[10] = 'B-GPE'
    ents[13] = 'B-PERSON'
    ents[14] = 'I-PERSON'
    tokens = en_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], heads=heads, deps=deps, tags=tags, ents=ents)
    assert len(doc) == 17
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            ent_type = max((w.ent_type_ for w in ent))
            attrs = {'lemma': ent.root.lemma_, 'ent_type': ent_type}
            retokenizer.merge(ent, attrs=attrs)
    assert len(doc) == 15