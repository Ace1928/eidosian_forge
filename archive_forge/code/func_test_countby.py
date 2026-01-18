from toolz import first, identity, countby, partitionby
def test_countby():
    assert countby(iseven, [1, 2, 3]) == {True: 1, False: 2}
    assert countby(len, ['cat', 'dog', 'mouse']) == {3: 2, 5: 1}
    assert countby(0, ('ab', 'ac', 'bc')) == {'a': 2, 'b': 1}