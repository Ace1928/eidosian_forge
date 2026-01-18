import sys
def filldoc(docdict, unindent_params=True):
    """ Return docstring decorator using docdict variable dictionary

    Parameters
    ----------
    docdict : dictionary
        dictionary containing name, docstring fragment pairs
    unindent_params : {False, True}, boolean, optional
        If True, strip common indentation from all parameters in
        docdict

    Returns
    -------
    decfunc : function
        decorator that applies dictionary to input function docstring

    """
    if unindent_params:
        docdict = unindent_dict(docdict)

    def decorate(f):
        f.__doc__ = docformat(f.__doc__, docdict)
        return f
    return decorate