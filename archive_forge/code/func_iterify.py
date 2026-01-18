def iterify(fun):
    """ Decorator to make a function apply
    to each item in a sequence, and return an iterator. """

    def f(seq):
        for x in seq:
            yield fun(x)
    return f