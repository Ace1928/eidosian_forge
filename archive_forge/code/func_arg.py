import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union, overload
from .._fields import MISSING_NONPROP
def arg(*, name: Optional[str]=None, metavar: Optional[str]=None, help: Optional[str]=None, aliases: Optional[Sequence[str]]=None, prefix_name: Optional[bool]=None, constructor: Optional[Union[Type, Callable]]=None, constructor_factory: Optional[Callable[[], Union[Type, Callable]]]=None) -> Any:
    """Returns a metadata object for fine-grained argument configuration with
    `typing.Annotated`. Should typically not be required.
    ```python
    x: Annotated[int, tyro.conf.arg(...)]
    ```

    Arguments:
        name: A new name for the argument in the CLI.
        metavar: Argument name in usage messages. The type is used by default.
        help: Helptext for this argument. The docstring is used by default.
        aliases: Aliases for this argument. All strings in the sequence should start
            with a hyphen (-). Aliases will _not_ currently be prefixed in a nested
            structure, and are not supported for positional arguments.
        prefix_name: Whether or not to prefix the name of the argument based on where
            it is in a nested structure. Arguments are prefixed by default.
        constructor: A constructor type or function. This should either be (a) a subtype
            of an argument's annotated type, or (b) a function with type-annotated
            inputs that returns an instance of the annotated type. This will be used in
            place of the argument's type for parsing arguments. No validation is done.
        constructor_factory: A function that returns a constructor type or function.
            Useful when the constructor isn't immediately available.

    Returns:
        Object to attach via `typing.Annotated[]`.
    """
    assert not (constructor is not None and constructor_factory is not None), '`constructor` and `constructor_factory` cannot both be set.'
    if aliases is not None:
        for alias in aliases:
            assert alias.startswith('-'), 'Argument alias needs to start with a hyphen!'
    return _ArgConfiguration(name=name, metavar=metavar, help=help, aliases=tuple(aliases) if aliases is not None else None, prefix_name=prefix_name, constructor_factory=constructor_factory if constructor is None else lambda: constructor)