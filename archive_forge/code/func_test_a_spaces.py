from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_a_spaces():
    assert md('foo <a href="http://google.com">Google</a> bar') == 'foo [Google](http://google.com) bar'
    assert md('foo<a href="http://google.com"> Google</a> bar') == 'foo [Google](http://google.com) bar'
    assert md('foo <a href="http://google.com">Google </a>bar') == 'foo [Google](http://google.com) bar'
    assert md('foo <a href="http://google.com"></a> bar') == 'foo  bar'