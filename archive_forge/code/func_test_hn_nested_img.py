from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_hn_nested_img():
    image_attributes_to_markdown = [('', '', ''), ("alt='Alt Text'", 'Alt Text', ''), ("alt='Alt Text' title='Optional title'", 'Alt Text', ' "Optional title"')]
    for image_attributes, markdown, title in image_attributes_to_markdown:
        assert md('<h3>A <img src="/path/to/img.jpg" ' + image_attributes + '/> B</h3>') == '### A ' + markdown + ' B\n\n'
        assert md('<h3>A <img src="/path/to/img.jpg" ' + image_attributes + '/> B</h3>', keep_inline_images_in=['h3']) == '### A ![' + markdown + '](/path/to/img.jpg' + title + ') B\n\n'