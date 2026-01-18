import sys
def inherit_docstring_from(cls):
    """
    This decorator modifies the decorated function's docstring by
    replacing occurrences of '%(super)s' with the docstring of the
    method of the same name from the class `cls`.

    If the decorated method has no docstring, it is simply given the
    docstring of `cls`s method.

    Parameters
    ----------
    cls : Python class or instance
        A class with a method with the same name as the decorated method.
        The docstring of the method in this class replaces '%(super)s' in the
        docstring of the decorated method.

    Returns
    -------
    f : function
        The decorator function that modifies the __doc__ attribute
        of its argument.

    Examples
    --------
    In the following, the docstring for Bar.func created using the
    docstring of `Foo.func`.

    >>> class Foo:
    ...     def func(self):
    ...         '''Do something useful.'''
    ...         return
    ...
    >>> class Bar(Foo):
    ...     @inherit_docstring_from(Foo)
    ...     def func(self):
    ...         '''%(super)s
    ...         Do it fast.
    ...         '''
    ...         return
    ...
    >>> b = Bar()
    >>> b.func.__doc__
    'Do something useful.
        Do it fast.
        '

    """

    def _doc(func):
        cls_docstring = getattr(cls, func.__name__).__doc__
        func_docstring = func.__doc__
        if func_docstring is None:
            func.__doc__ = cls_docstring
        else:
            new_docstring = func_docstring % dict(super=cls_docstring)
            func.__doc__ = new_docstring
        return func
    return _doc