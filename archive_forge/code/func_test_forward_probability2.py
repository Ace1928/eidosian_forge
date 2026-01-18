import pytest
from nltk.tag import hmm
def test_forward_probability2():
    from numpy.testing import assert_array_almost_equal
    model, states, symbols, seq = _wikipedia_example_hmm()
    fp = 2 ** model._forward_probability(seq)
    fp = (fp.T / fp.sum(axis=1)).T
    wikipedia_results = [[0.8182, 0.1818], [0.8834, 0.1166], [0.1907, 0.8093], [0.7308, 0.2692], [0.8673, 0.1327]]
    assert_array_almost_equal(wikipedia_results, fp, 4)