import re
from formencode.rewritingparser import RewritingParser, html_quote
def handle_textarea(self, attrs):
    name = self.get_attr(attrs, 'name')
    if self.prefix_error:
        self.write_marker(name)
    if self.error_class and self.errors.get(name):
        self.add_class(attrs, self.error_class)
    value = self.defaults.get(name, '')
    if value or self.force_defaults:
        self.write_tag('textarea', attrs)
        self.write_text(html_quote(value))
        self.write_text('</textarea>')
        self.skip_textarea = True
    self.in_textarea = True
    self.last_textarea_name = name
    self.add_key(name)