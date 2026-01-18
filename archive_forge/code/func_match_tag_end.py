import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_tag_end(self):
    match = self.match('\\</%[\\t ]*([^\\t ]+?)[\\t ]*>')
    if match:
        if not len(self.tag):
            raise exceptions.SyntaxException('Closing tag without opening tag: </%%%s>' % match.group(1), **self.exception_kwargs)
        elif self.tag[-1].keyword != match.group(1):
            raise exceptions.SyntaxException('Closing tag </%%%s> does not match tag: <%%%s>' % (match.group(1), self.tag[-1].keyword), **self.exception_kwargs)
        self.tag.pop()
        return True
    else:
        return False