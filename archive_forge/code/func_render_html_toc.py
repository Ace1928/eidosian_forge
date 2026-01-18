from ._base import DirectivePlugin
from ..toc import normalize_toc_item, render_toc_ul
def render_html_toc(renderer, title, collapse=False, **attrs):
    if not title:
        title = 'Table of Contents'
    toc = attrs['toc']
    content = render_toc_ul(attrs['toc'])
    html = '<details class="toc"'
    if not collapse:
        html += ' open'
    html += '>\n<summary>' + title + '</summary>\n'
    return html + content + '</details>\n'