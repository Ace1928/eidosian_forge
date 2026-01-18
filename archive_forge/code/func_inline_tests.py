from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def inline_tests(tag, markup):
    assert md(f'<{tag}>Hello</{tag}>') == f'{markup}Hello{markup}'
    assert md(f'foo <{tag}>Hello</{tag}> bar') == f'foo {markup}Hello{markup} bar'
    assert md(f'foo<{tag}> Hello</{tag}> bar') == f'foo {markup}Hello{markup} bar'
    assert md(f'foo <{tag}>Hello </{tag}>bar') == f'foo {markup}Hello{markup} bar'
    assert md(f'foo <{tag}></{tag}> bar') in ['foo  bar', 'foo bar']