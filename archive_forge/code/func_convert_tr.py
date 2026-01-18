from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_tr(self, el, text, convert_as_inline):
    cells = el.find_all(['td', 'th'])
    is_headrow = all([cell.name == 'th' for cell in cells])
    overline = ''
    underline = ''
    if is_headrow and (not el.previous_sibling):
        underline += '| ' + ' | '.join(['---'] * len(cells)) + ' |' + '\n'
    elif not el.previous_sibling and (el.parent.name == 'table' or (el.parent.name == 'tbody' and (not el.parent.previous_sibling))):
        overline += '| ' + ' | '.join([''] * len(cells)) + ' |' + '\n'
        overline += '| ' + ' | '.join(['---'] * len(cells)) + ' |' + '\n'
    return overline + '|' + text + '\n' + underline