from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
def bind_to_cls(self, cls: Any) -> Decorator[DecoratorInfoType]:
    """Bind the decorator to a class.

        Args:
            cls: the class.

        Returns:
            The new decorator instance.
        """
    return self.build(cls, cls_var_name=self.cls_var_name, shim=self.shim, info=self.info)