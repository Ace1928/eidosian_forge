from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_img():
    assert md('<img src="/path/to/img.jpg" alt="Alt text" title="Optional title" />') == '![Alt text](/path/to/img.jpg "Optional title")'
    assert md('<img src="/path/to/img.jpg" alt="Alt text" />') == '![Alt text](/path/to/img.jpg)'