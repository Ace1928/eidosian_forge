from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def process_tag(self, node, convert_as_inline, children_only=False):
    text = ''
    isHeading = html_heading_re.match(node.name) is not None
    isCell = node.name in ['td', 'th']
    convert_children_as_inline = convert_as_inline
    if not children_only and (isHeading or isCell):
        convert_children_as_inline = True

    def is_nested_node(el):
        return el and el.name in ['ol', 'ul', 'li', 'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th']
    if is_nested_node(node):
        for el in node.children:
            can_extract = not el.previous_sibling or not el.next_sibling or is_nested_node(el.previous_sibling) or is_nested_node(el.next_sibling)
            if isinstance(el, NavigableString) and six.text_type(el).strip() == '' and can_extract:
                el.extract()
    for el in node.children:
        if isinstance(el, Comment) or isinstance(el, Doctype):
            continue
        elif isinstance(el, NavigableString):
            text += self.process_text(el)
        else:
            text += self.process_tag(el, convert_children_as_inline)
    if not children_only:
        convert_fn = getattr(self, 'convert_%s' % node.name, None)
        if convert_fn and self.should_convert_tag(node.name):
            text = convert_fn(node, text, convert_as_inline)
    return text