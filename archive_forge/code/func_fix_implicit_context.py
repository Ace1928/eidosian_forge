from lib2to3 import fixer_base
from itertools import count
from lib2to3.fixer_util import (Assign, Comma, Call, Newline, Name,
from libfuturize.fixer_util import indentation, suitify, commatize
def fix_implicit_context(self, node, results):
    u"""
        Only example of the implicit context is
        a for loop, so only fix that.
        """
    pre, name, post, it = (results.get(n) for n in (u'pre', u'name', u'post', u'it'))
    pre = [n.clone() for n in pre if n.type == token.NAME]
    name.prefix = u' '
    post = [n.clone() for n in post if n.type == token.NAME]
    target = [n.clone() for n in commatize(pre + [name.clone()] + post)]
    target.append(Comma())
    source = it.clone()
    source.prefix = u''
    setup_line = Assign(Name(self.LISTNAME), Call(Name(u'list'), [Name(self.ITERNAME)]))
    power_line = Assign(target, assignment_source(len(pre), len(post), self.LISTNAME, self.ITERNAME))
    return (setup_line, power_line)