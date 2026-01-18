from yaql.language import specs
@specs.parameter('right', type(None), nullable=True)
@specs.parameter('left', nullable=False)
@specs.name('#operator_>=')
def left_gte_null(left, right):
    """:yaql:operator >=

    Returns true. This function is called when left is not null
    and right is null.

    :signature: left >= right
    :arg left: left operand
    :argType left: not null
    :arg right: right operand
    :argType right: null
    :returnType: boolean

    .. code:

        yaql> 1 >= null
        true
    """
    return True