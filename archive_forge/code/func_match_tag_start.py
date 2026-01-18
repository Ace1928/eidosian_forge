import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_tag_start(self):
    reg = '\n            \\<%     # opening tag\n\n            ([\\w\\.\\:]+)   # keyword\n\n            ((?:\\s+\\w+|\\s*=\\s*|"[^"]*?"|\'[^\']*?\'|\\s*,\\s*)*)  # attrname, = \\\n                                               #        sign, string expression\n                                               # comma is for backwards compat\n                                               # identified in #366\n\n            \\s*     # more whitespace\n\n            (/)?>   # closing\n\n        '
    match = self.match(reg, re.I | re.S | re.X)
    if not match:
        return False
    keyword, attr, isend = match.groups()
    self.keyword = keyword
    attributes = {}
    if attr:
        for att in re.findall('\\s*(\\w+)\\s*=\\s*(?:\'([^\']*)\'|\\"([^\\"]*)\\")', attr):
            key, val1, val2 = att
            text = val1 or val2
            text = text.replace('\r\n', '\n')
            attributes[key] = text
    self.append_node(parsetree.Tag, keyword, attributes)
    if isend:
        self.tag.pop()
    elif keyword == 'text':
        match = self.match('(.*?)(?=\\</%text>)', re.S)
        if not match:
            raise exceptions.SyntaxException('Unclosed tag: <%%%s>' % self.tag[-1].keyword, **self.exception_kwargs)
        self.append_node(parsetree.Text, match.group(1))
        return self.match_tag_end()
    return True