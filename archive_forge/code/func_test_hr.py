from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_hr():
    assert md('Hello<hr>World') == 'Hello\n\n---\n\nWorld'
    assert md('Hello<hr />World') == 'Hello\n\n---\n\nWorld'
    assert md('<p>Hello</p>\n<hr>\n<p>World</p>') == 'Hello\n\n\n\n\n---\n\n\nWorld\n\n'