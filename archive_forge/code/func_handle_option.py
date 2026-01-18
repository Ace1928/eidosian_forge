import re
from formencode.rewritingparser import RewritingParser, html_quote
def handle_option(self, attrs):
    assert self.in_select is not None, '<option> outside of <select> at %i:%i' % self.getpos()
    if self.in_select is not False:
        if self.force_defaults or self.in_select in self.defaults:
            if self.selected_multiple(self.defaults.get(self.in_select), self.get_attr(attrs, 'value', '')):
                self.set_attr(attrs, 'selected', 'selected')
                self.add_key(self.in_select)
            else:
                self.del_attr(attrs, 'selected')
    self.write_tag('option', attrs)
    self.skip_next = True