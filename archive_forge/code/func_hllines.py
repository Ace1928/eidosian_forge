from warnings import warn
def hllines(code, style):
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import HtmlFormatter
    except ImportError:
        raise ImportError("please install the 'pygments' package")
    pylex = PythonLexer()
    'Given a code string, return a list of html-highlighted lines'
    hf = HtmlFormatter(noclasses=True, style=style, nowrap=True)
    res = highlight(code, pylex, hf)
    return res.splitlines()