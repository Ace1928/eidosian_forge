import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union, overload
from .._fields import MISSING_NONPROP
def subcommand(name: Optional[str]=None, *, default: Any=MISSING_NONPROP, description: Optional[str]=None, prefix_name: bool=True, constructor: Optional[Union[Type, Callable]]=None, constructor_factory: Optional[Callable[[], Union[Type, Callable]]]=None) -> Any:
    """Returns a metadata object for configuring subcommands with `typing.Annotated`.
    Useful for aesthetics.

    Consider the standard approach for creating subcommands:

    ```python
    tyro.cli(
        Union[NestedTypeA, NestedTypeB]
    )
    ```

    This will create two subcommands: `nested-type-a` and `nested-type-b`.

    Annotating each type with `tyro.conf.subcommand()` allows us to override for
    each subcommand the (a) name, (b) defaults, (c) helptext, and (d) whether to prefix
    the name or not.

    ```python
    tyro.cli(
        Union[
            Annotated[
                NestedTypeA, subcommand("a", ...)
            ],
            Annotated[
                NestedTypeB, subcommand("b", ...)
            ],
        ]
    )
    ```

    Arguments:
        name: The name of the subcommand in the CLI.
        default: A default value for the subcommand, for struct-like types. (eg
             dataclasses)
        description: Description of this option to use in the helptext. Defaults to
            docstring.
        prefix_name: Whether to prefix the name of the subcommand based on where it
            is in a nested structure.
        constructor: A constructor type or function. This should either be (a) a subtype
            of an argument's annotated type, or (b) a function with type-annotated
            inputs that returns an instance of the annotated type. This will be used in
            place of the argument's type for parsing arguments. No validation is done.
        constructor_factory: A function that returns a constructor type or function.
            Useful when the constructor isn't immediately available.
    """
    assert not (constructor is not None and constructor_factory is not None), '`constructor` and `constructor_factory` cannot both be set.'
    return _SubcommandConfiguration(name, default, description, prefix_name, constructor_factory=constructor_factory if constructor is None else lambda: constructor)