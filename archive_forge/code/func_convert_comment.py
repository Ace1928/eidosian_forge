import re
from lxml import etree, html
@converter(Comment)
def convert_comment(bs_node, parent):
    res = html.HtmlComment(bs_node)
    if parent is not None:
        parent.append(res)
    return res