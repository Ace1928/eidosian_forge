import re
from formencode.rewritingparser import RewritingParser, html_quote
def handle_iferror(self, attrs):
    name = self.get_attr(attrs, 'name')
    assert name, 'Name attribute in <iferror> required at %i:%i' % self.getpos()
    notted = name.startswith('not ')
    if notted:
        name = name.split(None, 1)[1]
    self.in_error = name
    ok = self.errors.get(name)
    if notted:
        ok = not ok
    if not ok:
        self.skip_error = True
    self.skip_next = True