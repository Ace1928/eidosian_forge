from __future__ import annotations
import pathlib
from io import BytesIO, StringIO
from typing import Any, Dict, List, Tuple, TypedDict
import requests
from langchain_core.documents import Document
def split_text_from_file(self, file: Any) -> List[Document]:
    """Split HTML file

        Args:
            file: HTML file
        """
    try:
        from lxml import etree
    except ImportError as e:
        raise ImportError('Unable to import lxml, please install with `pip install lxml`.') from e
    parser = etree.HTMLParser(encoding='utf-8')
    tree = etree.parse(file, parser)
    xslt_path = pathlib.Path(__file__).parent / 'xsl/html_chunks_with_headers.xslt'
    xslt_tree = etree.parse(xslt_path)
    transform = etree.XSLT(xslt_tree)
    result = transform(tree)
    result_dom = etree.fromstring(str(result))
    header_filter = [header[0] for header in self.headers_to_split_on]
    header_mapping = dict(self.headers_to_split_on)
    ns_map = {'h': 'http://www.w3.org/1999/xhtml'}
    elements = []
    for element in result_dom.findall('*//*', ns_map):
        if element.findall("*[@class='headers']") or element.findall("*[@class='chunk']"):
            elements.append(ElementType(url=file, xpath=''.join([node.text or '' for node in element.findall("*[@class='xpath']", ns_map)]), content=''.join([node.text or '' for node in element.findall("*[@class='chunk']", ns_map)]), metadata={header_mapping[node.tag]: node.text or '' for node in filter(lambda x: x.tag in header_filter, element.findall("*[@class='headers']/*", ns_map))}))
    if not self.return_each_element:
        return self.aggregate_elements_to_chunks(elements)
    else:
        return [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in elements]