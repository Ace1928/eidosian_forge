import copy
import re
from collections import Counter as CollectionCounter, defaultdict, deque
from collections.abc import Callable, Hashable as CollectionsHashable, Iterable as CollectionsIterable
from typing import (
from typing_extensions import Annotated, Final
from . import errors as errors_
from .class_validators import Validator, make_generic_validator, prep_validators
from .error_wrappers import ErrorWrapper
from .errors import ConfigError, InvalidDiscriminator, MissingDiscriminator, NoneIsNotAllowedError
from .types import Json, JsonWrapper
from .typing import (
from .utils import (
from .validators import constant_validator, dict_validator, find_validators, validate_json
def prepare_discriminated_union_sub_fields(self) -> None:
    """
        Prepare the mapping <discriminator key> -> <ModelField> and update `sub_fields`
        Note that this process can be aborted if a `ForwardRef` is encountered
        """
    assert self.discriminator_key is not None
    if self.type_.__class__ is DeferredType:
        return
    assert self.sub_fields is not None
    sub_fields_mapping: Dict[str, 'ModelField'] = {}
    all_aliases: Set[str] = set()
    for sub_field in self.sub_fields:
        t = sub_field.type_
        if t.__class__ is ForwardRef:
            return
        alias, discriminator_values = get_discriminator_alias_and_values(t, self.discriminator_key)
        all_aliases.add(alias)
        for discriminator_value in discriminator_values:
            sub_fields_mapping[discriminator_value] = sub_field
    self.sub_fields_mapping = sub_fields_mapping
    self.discriminator_alias = get_unique_discriminator_alias(all_aliases, self.discriminator_key)