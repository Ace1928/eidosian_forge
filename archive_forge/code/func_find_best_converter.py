import re
from lxml import etree, html
def find_best_converter(node):
    for t in ordered_node_types:
        if isinstance(node, t):
            return converters[t]
    return None