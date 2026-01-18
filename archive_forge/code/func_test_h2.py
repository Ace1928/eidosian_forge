from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_h2():
    assert md('<h2>Hello</h2>') == 'Hello\n-----\n\n'