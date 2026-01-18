from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def parseCloseBracket(self, block):
    """
        Try to match close bracket against an opening in the delimiter
        stack. Add either a link or image, or a plain [ character,
        to block's children. If there is a matching delimiter,
        remove it from the delimiter stack.
        """
    title = None
    matched = False
    self.pos += 1
    startpos = self.pos
    opener = self.brackets
    if opener is None:
        block.append_child(text(']'))
        return True
    if not opener.get('active'):
        block.append_child(text(']'))
        self.removeBracket()
        return True
    is_image = opener.get('image')
    savepos = self.pos
    if self.peek() == '(':
        self.pos += 1
        self.spnl()
        dest = self.parseLinkDestination()
        if dest is not None and self.spnl():
            if re.search(reWhitespaceChar, self.subject[self.pos - 1]):
                title = self.parseLinkTitle()
            if self.spnl() and self.peek() == ')':
                self.pos += 1
                matched = True
        else:
            self.pos = savepos
    if not matched:
        beforelabel = self.pos
        n = self.parseLinkLabel()
        if n > 2:
            reflabel = self.subject[beforelabel:beforelabel + n]
        elif not opener.get('bracket_after'):
            reflabel = self.subject[opener.get('index'):startpos]
        if n == 0:
            self.pos = savepos
        if reflabel:
            link = self.refmap.get(normalize_reference(reflabel))
            if link:
                dest = link['destination']
                title = link['title']
                matched = True
    if matched:
        node = Node('image' if is_image else 'link', None)
        node.destination = dest
        node.title = title or ''
        tmp = opener.get('node').nxt
        while tmp:
            nxt = tmp.nxt
            tmp.unlink()
            node.append_child(tmp)
            tmp = nxt
        block.append_child(node)
        self.processEmphasis(opener.get('previousDelimiter'))
        self.removeBracket()
        opener.get('node').unlink()
        if not is_image:
            opener = self.brackets
            while opener is not None:
                if not opener.get('image'):
                    opener['active'] = False
                opener = opener.get('previous')
        return True
    else:
        self.removeBracket()
        self.pos = startpos
        block.append_child(text(']'))
        return True