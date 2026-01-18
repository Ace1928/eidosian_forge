from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_lang_callback():

    def callback(el):
        return el['class'][0] if el.has_attr('class') else None
    assert md('<pre class="python">test\n    foo\nbar</pre>', code_language_callback=callback) == '\n```python\ntest\n    foo\nbar\n```\n'
    assert md('<pre class="javascript"><code>test\n    foo\nbar</code></pre>', code_language_callback=callback) == '\n```javascript\ntest\n    foo\nbar\n```\n'
    assert md('<pre class="javascript"><code class="javascript">test\n    foo\nbar</code></pre>', code_language_callback=callback) == '\n```javascript\ntest\n    foo\nbar\n```\n'