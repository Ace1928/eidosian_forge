from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
def parse_layout(self, node):
    layout = Layout(show_empty=_to_bool(node.attrib.get('show_empty', False)), inline=_to_bool(node.attrib.get('inline', False)), inline_limit=int(node.attrib.get('inline_limit', 4)), inline_header=_to_bool(node.attrib.get('inline_header', True)), inline_alias=_to_bool(node.attrib.get('inline_alias', False)))
    order = []
    for child in node:
        tag, text = (child.tag, child.text)
        text = text.strip() if text else None
        if tag == 'Menuname' and text:
            order.append(['Menuname', text, _to_bool(child.attrib.get('show_empty', False)), _to_bool(child.attrib.get('inline', False)), int(child.attrib.get('inline_limit', 4)), _to_bool(child.attrib.get('inline_header', True)), _to_bool(child.attrib.get('inline_alias', False))])
        elif tag == 'Separator':
            order.append(['Separator'])
        elif tag == 'Filename' and text:
            order.append(['Filename', text])
        elif tag == 'Merge':
            order.append(['Merge', child.attrib.get('type', 'all')])
    layout.order = order
    return layout