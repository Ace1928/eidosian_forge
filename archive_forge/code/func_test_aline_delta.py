from nltk.metrics import aline
def test_aline_delta():
    """
    Test aline for computing the difference between two segments
    """
    assert aline.delta('p', 'q') == 20.0
    assert aline.delta('a', 'A') == 0.0