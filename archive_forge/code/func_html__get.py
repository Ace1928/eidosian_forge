import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def html__get(self):
    if not self.params.get('tag'):
        raise ValueError("You cannot get the HTML of %r until you set the 'tag' param'" % self)
    content = self._get_content()
    tag = '<%s' % self.params.get('tag')
    attrs = ' '.join(['%s="%s"' % (html_quote(n), html_quote(v)) for n, v in self._html_attrs()])
    if attrs:
        tag += ' ' + attrs
    tag += self._html_extra()
    if content is None:
        return tag + ' />'
    else:
        return '%s>%s</%s>' % (tag, content, self.params.get('tag'))