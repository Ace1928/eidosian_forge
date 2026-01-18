from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_a_with_title():
    text = md('<a href="http://google.com" title="The &quot;Goog&quot;">Google</a>')
    assert text == '[Google](http://google.com "The \\"Goog\\"")'
    assert md('<a href="https://google.com">https://google.com</a>', default_title=True) == '[https://google.com](https://google.com "https://google.com")'