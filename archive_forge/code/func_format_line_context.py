from .compat import escape
from .jsonify import encode
def format_line_context(filename, lineno, context=10):
    """
    Formats the the line context for error rendering.

    :param filename: the location of the file, within which the error occurred
    :param lineno: the offending line number
    :param context: number of lines of code to display before and after the
                    offending line.
    """
    with open(filename) as f:
        lines = f.readlines()
    lineno = lineno - 1
    if lineno > 0:
        start_lineno = max(lineno - context, 0)
        end_lineno = lineno + context
        lines = [escape(l, True) for l in lines[start_lineno:end_lineno]]
        i = lineno - start_lineno
        lines[i] = '<strong>%s</strong>' % lines[i]
    else:
        lines = [escape(l, True) for l in lines[:context]]
    msg = '<pre style="background-color:#ccc;padding:2em;">%s</pre>'
    return msg % ''.join(lines)