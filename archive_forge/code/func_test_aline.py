from nltk.metrics import aline
def test_aline():
    result = aline.align('θin', 'tenwis')
    expected = [[('θ', 't'), ('i', 'e'), ('n', 'n')]]
    assert result == expected
    result = aline.align('jo', 'ʒə')
    expected = [[('j', 'ʒ'), ('o', 'ə')]]
    assert result == expected
    result = aline.align('pematesiweni', 'pematesewen')
    expected = [[('p', 'p'), ('e', 'e'), ('m', 'm'), ('a', 'a'), ('t', 't'), ('e', 'e'), ('s', 's'), ('i', 'e'), ('w', 'w'), ('e', 'e'), ('n', 'n')]]
    assert result == expected
    result = aline.align('tuwθ', 'dentis')
    expected = [[('t', 't'), ('u', 'i'), ('w', '-'), ('θ', 's')]]
    assert result == expected