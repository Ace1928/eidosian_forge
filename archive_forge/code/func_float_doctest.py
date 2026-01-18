def float_doctest(sphinx_shell, args, input_lines, found, submitted):
    """
    Doctest which allow the submitted output to vary slightly from the input.

    Here is how it might appear in an rst file:

    .. code-block:: rst

       .. ipython::

          @doctest float
          In [1]: 0.1 + 0.2
          Out[1]: 0.3

    """
    import numpy as np
    if len(args) == 2:
        rtol = 1e-05
        atol = 1e-08
    else:
        try:
            rtol = float(args[2])
            atol = float(args[3])
        except IndexError as e:
            e = 'Both `rtol` and `atol` must be specified if either are specified: {0}'.format(args)
            raise IndexError(e) from e
    try:
        submitted = str_to_array(submitted)
        found = str_to_array(found)
    except:
        error = True
    else:
        found_isnan = np.isnan(found)
        submitted_isnan = np.isnan(submitted)
        error = not np.allclose(found_isnan, submitted_isnan)
        error |= not np.allclose(found[~found_isnan], submitted[~submitted_isnan], rtol=rtol, atol=atol)
    TAB = ' ' * 4
    directive = sphinx_shell.directive
    if directive is None:
        source = 'Unavailable'
        content = 'Unavailable'
    else:
        source = directive.state.document.current_source
        content = '\n'.join([TAB + line for line in directive.content])
    if error:
        e = 'doctest float comparison failure\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nOn input line(s):\n{TAB}{2}\n\nwe found output:\n{TAB}{3}\n\ninstead of the expected:\n{TAB}{4}\n\n'
        e = e.format(source, content, '\n'.join(input_lines), repr(found), repr(submitted), TAB=TAB)
        raise RuntimeError(e)