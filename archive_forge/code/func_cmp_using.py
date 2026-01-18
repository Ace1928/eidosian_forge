import functools
import types
from ._make import _make_ne
def cmp_using(eq=None, lt=None, le=None, gt=None, ge=None, require_same_type=True, class_name='Comparable'):
    """
    Create a class that can be passed into `attrs.field`'s ``eq``, ``order``,
    and ``cmp`` arguments to customize field comparison.

    The resulting class will have a full set of ordering methods if at least
    one of ``{lt, le, gt, ge}`` and ``eq``  are provided.

    :param Optional[callable] eq: `callable` used to evaluate equality of two
        objects.
    :param Optional[callable] lt: `callable` used to evaluate whether one
        object is less than another object.
    :param Optional[callable] le: `callable` used to evaluate whether one
        object is less than or equal to another object.
    :param Optional[callable] gt: `callable` used to evaluate whether one
        object is greater than another object.
    :param Optional[callable] ge: `callable` used to evaluate whether one
        object is greater than or equal to another object.

    :param bool require_same_type: When `True`, equality and ordering methods
        will return `NotImplemented` if objects are not of the same type.

    :param Optional[str] class_name: Name of class. Defaults to 'Comparable'.

    See `comparison` for more details.

    .. versionadded:: 21.1.0
    """
    body = {'__slots__': ['value'], '__init__': _make_init(), '_requirements': [], '_is_comparable_to': _is_comparable_to}
    num_order_functions = 0
    has_eq_function = False
    if eq is not None:
        has_eq_function = True
        body['__eq__'] = _make_operator('eq', eq)
        body['__ne__'] = _make_ne()
    if lt is not None:
        num_order_functions += 1
        body['__lt__'] = _make_operator('lt', lt)
    if le is not None:
        num_order_functions += 1
        body['__le__'] = _make_operator('le', le)
    if gt is not None:
        num_order_functions += 1
        body['__gt__'] = _make_operator('gt', gt)
    if ge is not None:
        num_order_functions += 1
        body['__ge__'] = _make_operator('ge', ge)
    type_ = types.new_class(class_name, (object,), {}, lambda ns: ns.update(body))
    if require_same_type:
        type_._requirements.append(_check_same_type)
    if 0 < num_order_functions < 4:
        if not has_eq_function:
            msg = 'eq must be define is order to complete ordering from lt, le, gt, ge.'
            raise ValueError(msg)
        type_ = functools.total_ordering(type_)
    return type_