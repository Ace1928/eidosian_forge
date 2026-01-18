import dataclasses
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
def apply_to_collections(data1: Optional[Any], data2: Optional[Any], dtype: Union[type, Any, Tuple[Union[type, Any]]], function: Callable, *args: Any, wrong_dtype: Optional[Union[type, Tuple[type]]]=None, **kwargs: Any) -> Any:
    """Zips two collections and applies a function to their items of a certain dtype.

    Args:
        data1: The first collection
        data2: The second collection
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection

    Raises:
        AssertionError:
            If sequence collections have different data sizes.

    """
    if data1 is None:
        if data2 is None:
            return None
        data1, data2 = (data2, None)
    elem_type = type(data1)
    if isinstance(data1, dtype) and data2 is not None and (wrong_dtype is None or not isinstance(data1, wrong_dtype)):
        return function(data1, data2, *args, **kwargs)
    if isinstance(data1, Mapping) and data2 is not None:
        zipped = {k: (data1[k], data2[k]) for k in data1.keys() | data2.keys()}
        return elem_type({k: apply_to_collections(*v, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs) for k, v in zipped.items()})
    is_namedtuple_ = is_namedtuple(data1)
    is_sequence = isinstance(data1, Sequence) and (not isinstance(data1, str))
    if (is_namedtuple_ or is_sequence) and data2 is not None:
        if len(data1) != len(data2):
            raise ValueError('Sequence collections have different sizes.')
        out = [apply_to_collections(v1, v2, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs) for v1, v2 in zip(data1, data2)]
        return elem_type(*out) if is_namedtuple_ else elem_type(out)
    if is_dataclass_instance(data1) and data2 is not None:
        if not is_dataclass_instance(data2):
            raise TypeError(f'Expected inputs to be dataclasses of the same type or to have identical fields but got input 1 of type {type(data1)} and input 2 of type {type(data2)}.')
        if not (len(dataclasses.fields(data1)) == len(dataclasses.fields(data2)) and all(map(lambda f1, f2: isinstance(f1, type(f2)), dataclasses.fields(data1), dataclasses.fields(data2)))):
            raise TypeError('Dataclasses fields do not match.')
        data = [data1, data2]
        fields: List[dict] = [{}, {}]
        memo: dict = {}
        for i in range(len(data)):
            for field in dataclasses.fields(data[i]):
                field_value = getattr(data[i], field.name)
                fields[i][field.name] = (field_value, field.init)
                if i == 0:
                    memo[id(field_value)] = field_value
        result = deepcopy(data1, memo=memo)
        for (field_name, (field_value1, field_init1)), (_, (field_value2, field_init2)) in zip(fields[0].items(), fields[1].items()):
            v = None
            if field_init1 and field_init2:
                v = apply_to_collections(field_value1, field_value2, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            if not field_init1 or not field_init2 or v is None:
                return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)
            try:
                setattr(result, field_name, v)
            except dataclasses.FrozenInstanceError as e:
                raise ValueError('A frozen dataclass was passed to `apply_to_collections` but this is not allowed.') from e
        return result
    return apply_to_collection(data1, dtype, function, *args, wrong_dtype=wrong_dtype, **kwargs)