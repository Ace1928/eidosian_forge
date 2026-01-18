import itertools
from yaql.language import contexts
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('name', yaqltypes.String())
@specs.parameter('func', yaqltypes.Lambda())
def def_(name, func, context):
    """:yaql:def

    Returns new context object with function name defined.

    :signature: def(name, func)
    :arg name: name of function
    :argType name: string
    :arg func: function to be stored under provided name
    :argType func: lambda
    :returnType: context object

    .. code::

        yaql> def(sq, $*$) -> [1, 2, 3].select(sq($))
        [1, 4, 9]
    """

    @specs.name(name)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    context.register_function(wrapper)
    return context