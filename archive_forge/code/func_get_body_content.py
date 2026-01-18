import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def get_body_content(self):
    """
        Returns content of BODY element for this HTML document. Content will be of type 'str' (Python 2)
        or 'bytes' (Python 3).

        :Returns:
          Returns content of this document.
        """
    try:
        html_tree = parse_html_string(self.content)
    except:
        return ''
    html_root = html_tree.getroottree()
    if len(html_root.find('body')) != 0:
        body = html_tree.find('body')
        tree_str = etree.tostring(body, pretty_print=True, encoding='utf-8', xml_declaration=False)
        if tree_str.startswith(six.b('<body>')):
            n = tree_str.rindex(six.b('</body>'))
            return tree_str[6:n]
        return tree_str
    return ''