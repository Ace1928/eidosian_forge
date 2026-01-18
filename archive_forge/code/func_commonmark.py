from __future__ import absolute_import, unicode_literals
from commonmark.blocks import Parser
from commonmark.dump import dumpAST, dumpJSON
from commonmark.render.html import HtmlRenderer
from commonmark.render.rst import ReStructuredTextRenderer
def commonmark(text, format='html'):
    """Render CommonMark into HTML, JSON or AST
    Optional keyword arguments:
    format:     'html' (default), 'json' or 'ast'

    >>> commonmark("*hello!*")
    '<p><em>hello</em></p>\\n'
    """
    parser = Parser()
    ast = parser.parse(text)
    if format not in ['html', 'json', 'ast', 'rst']:
        raise ValueError("format must be 'html', 'json' or 'ast'")
    if format == 'html':
        renderer = HtmlRenderer()
        return renderer.render(ast)
    if format == 'json':
        return dumpJSON(ast)
    if format == 'ast':
        return dumpAST(ast)
    if format == 'rst':
        renderer = ReStructuredTextRenderer()
        return renderer.render(ast)