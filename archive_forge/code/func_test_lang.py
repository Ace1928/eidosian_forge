from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_lang():
    assert md('<pre>test\n    foo\nbar</pre>', code_language='python') == '\n```python\ntest\n    foo\nbar\n```\n'
    assert md('<pre><code>test\n    foo\nbar</code></pre>', code_language='javascript') == '\n```javascript\ntest\n    foo\nbar\n```\n'