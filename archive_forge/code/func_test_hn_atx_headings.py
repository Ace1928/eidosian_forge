from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_hn_atx_headings():
    assert md('<h1>Hello</h1>', heading_style=ATX) == '# Hello\n\n'
    assert md('<h2>Hello</h2>', heading_style=ATX) == '## Hello\n\n'