from markdownify import markdownify as md
def test_do_not_strip():
    text = md('<a href="https://github.com/matthewwithanm">Some Text</a>', strip=[])
    assert text == '[Some Text](https://github.com/matthewwithanm)'