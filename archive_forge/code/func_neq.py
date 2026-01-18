from yaql.language import specs
@specs.name('*not_equal')
def neq(left, right):
    """:yaql:operator !=

    Returns true if left and right are not equal, false otherwise.

    It is system function and can be used to override behavior
    of comparison between objects.
    """
    return left != right