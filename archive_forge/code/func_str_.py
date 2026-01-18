import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('value', nullable=True)
def str_(value):
    """:yaql:str

    Returns a string representation of the value.

    :signature: str(value)
    :arg value: value to be evaluated to string
    :argType value: any
    :returnType: string

    .. code::

        yaql> str(["abc", "de"])
        "(u'abc', u'd')"
        yaql> str(123)
        "123"
    """
    if value is None:
        return 'null'
    elif value is True:
        return 'true'
    elif value is False:
        return 'false'
    else:
        return str(value)