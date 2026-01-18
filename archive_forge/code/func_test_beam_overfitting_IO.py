import pytest
from numpy.testing import assert_equal
from thinc.api import Adam
from spacy import registry, util
from spacy.attrs import DEP, NORM
from spacy.lang.en import English
from spacy.pipeline import DependencyParser
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from ..util import apply_transition_sequence, make_tempdir
def test_beam_overfitting_IO():
    nlp = English()
    beam_width = 16
    beam_density = 0.0001
    config = {'beam_width': beam_width, 'beam_density': beam_density}
    parser = nlp.add_pipe('beam_parser', config=config)
    train_examples = []
    for text, annotations in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for dep in annotations.get('deps', []):
            parser.add_label(dep)
    optimizer = nlp.initialize()
    for i in range(150):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['beam_parser'] < 0.0001
    test_text = 'I like securities.'
    docs = [nlp.make_doc(test_text)]
    beams = parser.predict(docs)
    head_scores, label_scores = parser.scored_parses(beams)
    head_scores = head_scores[0]
    label_scores = label_scores[0]
    assert label_scores[0, 'nsubj'] == pytest.approx(1.0, abs=eps)
    assert label_scores[0, 'dobj'] == pytest.approx(0.0, abs=eps)
    assert label_scores[0, 'punct'] == pytest.approx(0.0, abs=eps)
    assert label_scores[2, 'nsubj'] == pytest.approx(0.0, abs=eps)
    assert label_scores[2, 'dobj'] == pytest.approx(1.0, abs=eps)
    assert label_scores[2, 'punct'] == pytest.approx(0.0, abs=eps)
    assert label_scores[3, 'nsubj'] == pytest.approx(0.0, abs=eps)
    assert label_scores[3, 'dobj'] == pytest.approx(0.0, abs=eps)
    assert label_scores[3, 'punct'] == pytest.approx(1.0, abs=eps)
    assert head_scores[0, 0] == pytest.approx(0.0, abs=eps)
    assert head_scores[0, 1] == pytest.approx(1.0, abs=eps)
    assert head_scores[0, 2] == pytest.approx(0.0, abs=eps)
    assert head_scores[2, 0] == pytest.approx(0.0, abs=eps)
    assert head_scores[2, 1] == pytest.approx(1.0, abs=eps)
    assert head_scores[2, 2] == pytest.approx(0.0, abs=eps)
    assert head_scores[3, 0] == pytest.approx(0.0, abs=eps)
    assert head_scores[3, 1] == pytest.approx(1.0, abs=eps)
    assert head_scores[3, 2] == pytest.approx(0.0, abs=eps)
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        docs2 = [nlp2.make_doc(test_text)]
        parser2 = nlp2.get_pipe('beam_parser')
        beams2 = parser2.predict(docs2)
        head_scores2, label_scores2 = parser2.scored_parses(beams2)
        head_scores2 = head_scores2[0]
        label_scores2 = label_scores2[0]
        assert label_scores2[0, 'nsubj'] == pytest.approx(1.0, abs=eps)
        assert label_scores2[0, 'dobj'] == pytest.approx(0.0, abs=eps)
        assert label_scores2[0, 'punct'] == pytest.approx(0.0, abs=eps)
        assert label_scores2[2, 'nsubj'] == pytest.approx(0.0, abs=eps)
        assert label_scores2[2, 'dobj'] == pytest.approx(1.0, abs=eps)
        assert label_scores2[2, 'punct'] == pytest.approx(0.0, abs=eps)
        assert label_scores2[3, 'nsubj'] == pytest.approx(0.0, abs=eps)
        assert label_scores2[3, 'dobj'] == pytest.approx(0.0, abs=eps)
        assert label_scores2[3, 'punct'] == pytest.approx(1.0, abs=eps)
        assert head_scores2[0, 0] == pytest.approx(0.0, abs=eps)
        assert head_scores2[0, 1] == pytest.approx(1.0, abs=eps)
        assert head_scores2[0, 2] == pytest.approx(0.0, abs=eps)
        assert head_scores2[2, 0] == pytest.approx(0.0, abs=eps)
        assert head_scores2[2, 1] == pytest.approx(1.0, abs=eps)
        assert head_scores2[2, 2] == pytest.approx(0.0, abs=eps)
        assert head_scores2[3, 0] == pytest.approx(0.0, abs=eps)
        assert head_scores2[3, 1] == pytest.approx(1.0, abs=eps)
        assert head_scores2[3, 2] == pytest.approx(0.0, abs=eps)