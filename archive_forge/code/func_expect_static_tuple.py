from .. import debug
def expect_static_tuple(obj):
    """Check if the passed object is a StaticTuple.

    Cast it if necessary, but if the 'static_tuple' debug flag is set, raise an
    error instead.

    As apis are improved, we will probably eventually stop calling this as it
    adds overhead we shouldn't need.
    """
    if 'static_tuple' not in debug.debug_flags:
        return StaticTuple.from_sequence(obj)
    if not isinstance(obj, StaticTuple):
        raise TypeError('We expected a StaticTuple not a {}'.format(type(obj)))
    return obj