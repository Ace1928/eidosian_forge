import io
import mimetypes
from lxml import etree
def get_pages(item):
    body = parse_html_string(item.get_body_content())
    pages = []
    for elem in body.iter():
        if 'epub:type' in elem.attrib:
            if elem.get('id') is not None:
                _text = None
                if elem.text is not None and elem.text.strip() != '':
                    _text = elem.text.strip()
                if _text is None:
                    _text = elem.get('aria-label')
                if _text is None:
                    _text = get_headers(elem)
                pages.append((item.get_name(), elem.get('id'), _text or elem.get('id')))
    return pages