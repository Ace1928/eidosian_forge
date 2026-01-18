from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_blockquote_nested():
    text = md('<blockquote>And she was like <blockquote>Hello</blockquote></blockquote>')
    assert text == '\n> And she was like \n> > Hello\n> \n> \n\n'