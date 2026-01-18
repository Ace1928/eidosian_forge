from markdownify import markdownify as md
def test_inline_ul():
    assert md('<p>foo</p><ul><li>a</li><li>b</li></ul><p>bar</p>') == 'foo\n\n* a\n* b\n\nbar\n\n'