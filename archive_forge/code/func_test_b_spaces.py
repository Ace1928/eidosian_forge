from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_b_spaces():
    assert md('foo <b>Hello</b> bar') == 'foo **Hello** bar'
    assert md('foo<b> Hello</b> bar') == 'foo **Hello** bar'
    assert md('foo <b>Hello </b>bar') == 'foo **Hello** bar'
    assert md('foo <b></b> bar') == 'foo  bar'