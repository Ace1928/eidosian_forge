from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_hn_nested_simple_tag():
    tag_to_markdown = [('strong', '**strong**'), ('b', '**b**'), ('em', '*em*'), ('i', '*i*'), ('p', 'p'), ('a', 'a'), ('div', 'div'), ('blockquote', 'blockquote')]
    for tag, markdown in tag_to_markdown:
        assert md('<h3>A <' + tag + '>' + tag + '</' + tag + '> B</h3>') == '### A ' + markdown + ' B\n\n'
    assert md('<h3>A <br>B</h3>', heading_style=ATX) == '### A B\n\n'