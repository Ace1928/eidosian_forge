from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_br():
    assert md('a<br />b<br />c') == 'a  \nb  \nc'
    assert md('a<br />b<br />c', newline_style=BACKSLASH) == 'a\\\nb\\\nc'