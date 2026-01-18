from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_blockquote():
    assert md('<blockquote>Hello</blockquote>') == '\n> Hello\n\n'