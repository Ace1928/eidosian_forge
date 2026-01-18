from ._base import DirectivePlugin
def render_admonition(self, text, name, **attrs):
    html = '<section class="admonition ' + name
    _cls = attrs.get('class')
    if _cls:
        html += ' ' + _cls
    return html + '">\n' + text + '</section>\n'