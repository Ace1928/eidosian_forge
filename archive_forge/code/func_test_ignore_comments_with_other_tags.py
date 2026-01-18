from markdownify import markdownify as md
def test_ignore_comments_with_other_tags():
    text = md("<!-- This is a comment --><a href='http://example.com/'>example link</a>")
    assert text == '[example link](http://example.com/)'