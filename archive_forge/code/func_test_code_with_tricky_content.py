from markdownify import markdownify as md
def test_code_with_tricky_content():
    assert md('<code>></code>') == '`>`'
    assert md('<code>/home/</code><b>username</b>') == '`/home/`**username**'
    assert md('First line <code>blah blah<br />blah blah</code> second line') == 'First line `blah blah  \nblah blah` second line'