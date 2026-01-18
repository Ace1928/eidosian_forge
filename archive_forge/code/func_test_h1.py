from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_h1():
    assert md('<h1>Hello</h1>') == 'Hello\n=====\n\n'