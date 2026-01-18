from markdownify import markdownify as md
def test_chomp():
    assert md(' <b></b> ') == '  '
    assert md(' <b> </b> ') == '  '
    assert md(' <b>  </b> ') == '  '
    assert md(' <b>   </b> ') == '  '
    assert md(' <b>s </b> ') == ' **s**  '
    assert md(' <b> s</b> ') == '  **s** '
    assert md(' <b> s </b> ') == '  **s**  '
    assert md(' <b>  s  </b> ') == '  **s**  '