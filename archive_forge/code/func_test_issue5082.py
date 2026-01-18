import gc
import numpy
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.lang.en.syntax_iterators import noun_chunks
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList, get_arg_names, make_tempdir
from spacy.vocab import Vocab
@pytest.mark.issue(5082)
def test_issue5082():
    nlp = English()
    vocab = nlp.vocab
    array1 = numpy.asarray([0.1, 0.5, 0.8], dtype=numpy.float32)
    array2 = numpy.asarray([-0.2, -0.6, -0.9], dtype=numpy.float32)
    array3 = numpy.asarray([0.3, -0.1, 0.7], dtype=numpy.float32)
    array4 = numpy.asarray([0.5, 0, 0.3], dtype=numpy.float32)
    array34 = numpy.asarray([0.4, -0.05, 0.5], dtype=numpy.float32)
    vocab.set_vector('I', array1)
    vocab.set_vector('like', array2)
    vocab.set_vector('David', array3)
    vocab.set_vector('Bowie', array4)
    text = 'I like David Bowie'
    patterns = [{'label': 'PERSON', 'pattern': [{'LOWER': 'david'}, {'LOWER': 'bowie'}]}]
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    parsed_vectors_1 = [t.vector for t in nlp(text)]
    assert len(parsed_vectors_1) == 4
    ops = get_current_ops()
    numpy.testing.assert_array_equal(ops.to_numpy(parsed_vectors_1[0]), array1)
    numpy.testing.assert_array_equal(ops.to_numpy(parsed_vectors_1[1]), array2)
    numpy.testing.assert_array_equal(ops.to_numpy(parsed_vectors_1[2]), array3)
    numpy.testing.assert_array_equal(ops.to_numpy(parsed_vectors_1[3]), array4)
    nlp.add_pipe('merge_entities')
    parsed_vectors_2 = [t.vector for t in nlp(text)]
    assert len(parsed_vectors_2) == 3
    numpy.testing.assert_array_equal(ops.to_numpy(parsed_vectors_2[0]), array1)
    numpy.testing.assert_array_equal(ops.to_numpy(parsed_vectors_2[1]), array2)
    numpy.testing.assert_array_equal(ops.to_numpy(parsed_vectors_2[2]), array34)