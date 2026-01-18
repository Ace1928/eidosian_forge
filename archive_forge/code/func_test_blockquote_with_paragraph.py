from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_blockquote_with_paragraph():
    assert md('<blockquote>Hello</blockquote><p>handsome</p>') == '\n> Hello\n\nhandsome\n\n'