import pickle
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.lookups import Lookups
from ..util import make_tempdir
def test_lemmatizer_serialize(nlp):
    lemmatizer = nlp.add_pipe('lemmatizer', config={'mode': 'rule'})
    nlp.initialize()

    def cope_lookups():
        lookups = Lookups()
        lookups.add_table('lemma_lookup', {'cope': 'cope', 'coped': 'cope'})
        lookups.add_table('lemma_index', {'verb': ('cope', 'cop')})
        lookups.add_table('lemma_exc', {'verb': {'coping': ('cope',)}})
        lookups.add_table('lemma_rules', {'verb': [['ing', '']]})
        return lookups
    nlp2 = English()
    lemmatizer2 = nlp2.add_pipe('lemmatizer', config={'mode': 'rule'})
    lemmatizer2.initialize(lookups=cope_lookups())
    lemmatizer2.from_bytes(lemmatizer.to_bytes())
    assert lemmatizer.to_bytes() == lemmatizer2.to_bytes()
    assert lemmatizer.lookups.tables == lemmatizer2.lookups.tables
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
    doc2 = nlp2.make_doc('coping')
    doc2[0].pos_ = 'VERB'
    assert doc2[0].lemma_ == ''
    doc2 = lemmatizer2(doc2)
    assert doc2[0].text == 'coping'
    assert doc2[0].lemma_ == 'cope'
    pickle.dumps(lemmatizer2)