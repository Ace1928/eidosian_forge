from .roundtrip import dedent
def test_multiple_dedent(self):
    x = dedent(dedent('\n        123\n        '))
    assert x == '123\n'