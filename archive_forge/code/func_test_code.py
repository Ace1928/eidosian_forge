from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_code():
    inline_tests('code', '`')
    assert md('<code>this_should_not_escape</code>') == '`this_should_not_escape`'