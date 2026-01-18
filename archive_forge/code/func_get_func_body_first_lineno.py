import ast
def get_func_body_first_lineno(pyfunc):
    """
    Look up the first line of function body using the file in
    ``pyfunc.__code__.co_filename``.

    Returns
    -------
    lineno : int; or None
        The first line number of the function body; or ``None`` if the first
        line cannot be determined.
    """
    co = pyfunc.__code__
    try:
        with open(co.co_filename) as fin:
            file_content = fin.read()
    except (FileNotFoundError, OSError):
        return
    else:
        tree = ast.parse(file_content)
        finder = FindDefFirstLine(co)
        finder.visit(tree)
        return finder.first_stmt_line