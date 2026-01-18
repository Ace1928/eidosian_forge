import io
import mimetypes
from lxml import etree
def create_pagebreak(pageref, label=None, html=True):
    from ebooklib.epub import NAMESPACES
    pageref_attributes = {'{%s}type' % NAMESPACES['EPUB']: 'pagebreak', 'title': u'{}'.format(pageref), 'id': u'{}'.format(pageref)}
    pageref_elem = etree.Element('span', pageref_attributes, nsmap={'epub': NAMESPACES['EPUB']})
    if label:
        pageref_elem.text = label
    if html:
        return etree.tostring(pageref_elem, encoding='unicode')
    return pageref_elem