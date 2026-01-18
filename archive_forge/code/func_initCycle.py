def initCycle(container, seq_func, n=1):
    """Call the function *container* with a generator function corresponding
    to the calling *n* times the functions present in *seq_func*.

    :param container: The type to put in the data from func.
    :param seq_func: A list of function objects to be called in order to
                     fill the container.
    :param n: Number of times to iterate through the list of functions.
    :returns: An instance of the container filled with data from the
              returned by the functions.

    This helper function can be used in conjunction with a Toolbox
    to register a generator of filled containers, as individuals or
    population.

        >>> func_seq = [lambda:1 , lambda:'a', lambda:3]
        >>> initCycle(list, func_seq, n=2)
        [1, 'a', 3, 1, 'a', 3]

    See the :ref:`funky` tutorial for an example.
    """
    return container((func() for _ in range(n) for func in seq_func))