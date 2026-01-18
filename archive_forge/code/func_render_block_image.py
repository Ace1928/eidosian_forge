import re
from ._base import DirectivePlugin
from ..util import escape as escape_text, escape_url
def render_block_image(self, src: str, alt=None, width=None, height=None, **attrs):
    img = '<img src="' + src + '"'
    style = ''
    if alt:
        img += ' alt="' + escape_text(alt) + '"'
    if width:
        if width.isdigit():
            img += ' width="' + width + '"'
        else:
            style += 'width:' + width + ';'
    if height:
        if height.isdigit():
            img += ' height="' + height + '"'
        else:
            style += 'height:' + height + ';'
    if style:
        img += ' style="' + escape_text(style) + '"'
    img += ' />'
    _cls = 'block-image'
    align = attrs.get('align')
    if align:
        _cls += ' align-' + align
    target = attrs.get('target')
    if target:
        href = escape_text(self.safe_url(target))
        outer = '<a class="' + _cls + '" href="' + href + '">'
        return outer + img + '</a>\n'
    else:
        return '<div class="' + _cls + '">' + img + '</div>\n'