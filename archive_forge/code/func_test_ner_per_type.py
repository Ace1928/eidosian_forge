import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
def test_ner_per_type(en_vocab):
    scorer = Scorer()
    examples = []
    for input_, annot in test_ner_cardinal:
        doc = Doc(en_vocab, words=input_.split(' '), ents=['B-CARDINAL', 'O', 'B-CARDINAL'])
        entities = offsets_to_biluo_tags(doc, annot['entities'])
        example = Example.from_dict(doc, {'entities': entities})
        example.predicted[1].is_sent_start = False
        example.reference[1].is_sent_start = False
        examples.append(example)
    results = scorer.score(examples)
    assert results['ents_p'] == 1.0
    assert results['ents_r'] == 1.0
    assert results['ents_f'] == 1.0
    assert results['ents_per_type']['CARDINAL']['p'] == 1.0
    assert results['ents_per_type']['CARDINAL']['r'] == 1.0
    assert results['ents_per_type']['CARDINAL']['f'] == 1.0
    scorer = Scorer()
    examples = []
    for input_, annot in test_ner_apple:
        doc = Doc(en_vocab, words=input_.split(' '), ents=['B-ORG', 'O', 'O', 'O', 'O', 'B-GPE', 'B-ORG', 'O', 'O', 'O'])
        entities = offsets_to_biluo_tags(doc, annot['entities'])
        example = Example.from_dict(doc, {'entities': entities})
        example.predicted[1].is_sent_start = False
        example.reference[1].is_sent_start = False
        examples.append(example)
    results = scorer.score(examples)
    assert results['ents_p'] == approx(0.6666666)
    assert results['ents_r'] == approx(0.6666666)
    assert results['ents_f'] == approx(0.6666666)
    assert 'GPE' in results['ents_per_type']
    assert 'MONEY' in results['ents_per_type']
    assert 'ORG' in results['ents_per_type']
    assert results['ents_per_type']['GPE']['p'] == 1.0
    assert results['ents_per_type']['GPE']['r'] == 1.0
    assert results['ents_per_type']['GPE']['f'] == 1.0
    assert results['ents_per_type']['MONEY']['p'] == 0
    assert results['ents_per_type']['MONEY']['r'] == 0
    assert results['ents_per_type']['MONEY']['f'] == 0
    assert results['ents_per_type']['ORG']['p'] == 0.5
    assert results['ents_per_type']['ORG']['r'] == 1.0
    assert results['ents_per_type']['ORG']['f'] == approx(0.6666666)