import pytest
from thinc.api import Config
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.span_finder import span_finder_default_config
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import fix_random_seed, make_tempdir, registry
@pytest.mark.parametrize('tokens_predicted, tokens_reference, reference_truths', [(['Mon', '.', '-', 'June', '16'], ['Mon.', '-', 'June', '16'], [(0, 0), (0, 0), (0, 0), (1, 1), (0, 0)]), (['Mon.', '-', 'J', 'une', '16'], ['Mon.', '-', 'June', '16'], [(0, 0), (0, 0), (1, 0), (0, 1), (0, 0)]), (['Mon', '.', '-', 'June', '16'], ['Mon.', '-', 'June', '1', '6'], [(0, 0), (0, 0), (0, 0), (1, 1), (0, 0)]), (['Mon.', '-J', 'un', 'e 16'], ['Mon.', '-', 'June', '16'], [(0, 0), (0, 0), (0, 0), (0, 0)]), pytest.param(['Mon.-June', '16'], ['Mon.', '-', 'June', '16'], [(0, 1), (0, 0)]), pytest.param(['Mon.-', 'June', '16'], ['Mon.', '-', 'J', 'une', '16'], [(0, 0), (1, 1), (0, 0)]), pytest.param(['Mon.-', 'June 16'], ['Mon.', '-', 'June', '16'], [(0, 0), (1, 0)])])
def test_loss_alignment_example(tokens_predicted, tokens_reference, reference_truths):
    nlp = Language()
    predicted = Doc(nlp.vocab, words=tokens_predicted, spaces=[False] * len(tokens_predicted))
    reference = Doc(nlp.vocab, words=tokens_reference, spaces=[False] * len(tokens_reference))
    example = Example(predicted, reference)
    example.reference.spans[SPANS_KEY] = [example.reference.char_span(5, 9)]
    span_finder = nlp.add_pipe('span_finder', config={'spans_key': SPANS_KEY})
    nlp.initialize()
    ops = span_finder.model.ops
    if predicted.text != reference.text:
        with pytest.raises(ValueError, match='must match between reference and predicted'):
            span_finder._get_aligned_truth_scores([example], ops)
        return
    truth_scores, masks = span_finder._get_aligned_truth_scores([example], ops)
    assert len(truth_scores) == len(tokens_predicted)
    ops.xp.testing.assert_array_equal(truth_scores, ops.xp.asarray(reference_truths))