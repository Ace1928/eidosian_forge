from yaql.language import specs
@specs.parameter('left', type(None), nullable=True)
@specs.parameter('right', type(None), nullable=True)
@specs.name('#operator_>')
def null_gt_null(left, right):
    """:yaql:operator >

    Returns false. This function is called when left and right are null.

    :signature: left > right
    :arg left: left operand
    :argType left: null
    :arg right: right operand
    :argType right: null
    :returnType: boolean

    .. code:

        yaql> null > null
        false
    """
    return False