from uritemplate.orderedset import OrderedSet
from uritemplate.template import URITemplate
def partial(uri, var_dict=None, **kwargs):
    """Partially expand the template with the given parameters.

    If all of the parameters for the template are not given, return a
    partially expanded template.

    :param dict var_dict: Optional dictionary with variables and values
    :param kwargs: Alternative way to pass arguments
    :returns: :class:`URITemplate`

    Example::

        t = URITemplate('https://api.github.com{/end}')
        t.partial()  # => URITemplate('https://api.github.com{/end}')

    """
    return URITemplate(uri).partial(var_dict, **kwargs)