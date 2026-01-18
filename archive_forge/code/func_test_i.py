from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_i():
    assert md('<i>Hello</i>') == '*Hello*'