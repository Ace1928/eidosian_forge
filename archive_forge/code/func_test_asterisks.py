from markdownify import markdownify as md
def test_asterisks():
    assert md('*hey*dude*') == '\\*hey\\*dude\\*'
    assert md('*hey*dude*', escape_asterisks=False) == '*hey*dude*'