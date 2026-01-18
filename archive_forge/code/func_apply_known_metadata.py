from __future__ import annotations
from collections import defaultdict
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable
from pydantic_core import CoreSchema, PydanticCustomError, to_jsonable_python
from pydantic_core import core_schema as cs
from ._fields import PydanticMetadata
def apply_known_metadata(annotation: Any, schema: CoreSchema) -> CoreSchema | None:
    """Apply `annotation` to `schema` if it is an annotation we know about (Gt, Le, etc.).
    Otherwise return `None`.

    This does not handle all known annotations. If / when it does, it can always
    return a CoreSchema and return the unmodified schema if the annotation should be ignored.

    Assumes that GroupedMetadata has already been expanded via `expand_grouped_metadata`.

    Args:
        annotation: The annotation.
        schema: The schema.

    Returns:
        An updated schema with annotation if it is an annotation we know about, `None` otherwise.

    Raises:
        PydanticCustomError: If `Predicate` fails.
    """
    import annotated_types as at
    from . import _validators
    schema = schema.copy()
    schema_update, other_metadata = collect_known_metadata([annotation])
    schema_type = schema['type']
    for constraint, value in schema_update.items():
        if constraint not in CONSTRAINTS_TO_ALLOWED_SCHEMAS:
            raise ValueError(f'Unknown constraint {constraint}')
        allowed_schemas = CONSTRAINTS_TO_ALLOWED_SCHEMAS[constraint]
        if schema_type in allowed_schemas:
            if constraint == 'union_mode' and schema_type == 'union':
                schema['mode'] = value
            else:
                schema[constraint] = value
            continue
        if constraint == 'allow_inf_nan' and value is False:
            return cs.no_info_after_validator_function(_validators.forbid_inf_nan_check, schema)
        elif constraint == 'pattern':
            return cs.chain_schema([schema, cs.str_schema(pattern=value)])
        elif constraint == 'gt':
            s = cs.no_info_after_validator_function(partial(_validators.greater_than_validator, gt=value), schema)
            add_js_update_schema(s, lambda: {'gt': as_jsonable_value(value)})
            return s
        elif constraint == 'ge':
            return cs.no_info_after_validator_function(partial(_validators.greater_than_or_equal_validator, ge=value), schema)
        elif constraint == 'lt':
            return cs.no_info_after_validator_function(partial(_validators.less_than_validator, lt=value), schema)
        elif constraint == 'le':
            return cs.no_info_after_validator_function(partial(_validators.less_than_or_equal_validator, le=value), schema)
        elif constraint == 'multiple_of':
            return cs.no_info_after_validator_function(partial(_validators.multiple_of_validator, multiple_of=value), schema)
        elif constraint == 'min_length':
            s = cs.no_info_after_validator_function(partial(_validators.min_length_validator, min_length=value), schema)
            add_js_update_schema(s, lambda: {'minLength': as_jsonable_value(value)})
            return s
        elif constraint == 'max_length':
            s = cs.no_info_after_validator_function(partial(_validators.max_length_validator, max_length=value), schema)
            add_js_update_schema(s, lambda: {'maxLength': as_jsonable_value(value)})
            return s
        elif constraint == 'strip_whitespace':
            return cs.chain_schema([schema, cs.str_schema(strip_whitespace=True)])
        elif constraint == 'to_lower':
            return cs.chain_schema([schema, cs.str_schema(to_lower=True)])
        elif constraint == 'to_upper':
            return cs.chain_schema([schema, cs.str_schema(to_upper=True)])
        elif constraint == 'min_length':
            return cs.no_info_after_validator_function(partial(_validators.min_length_validator, min_length=annotation.min_length), schema)
        elif constraint == 'max_length':
            return cs.no_info_after_validator_function(partial(_validators.max_length_validator, max_length=annotation.max_length), schema)
        else:
            raise RuntimeError(f'Unable to apply constraint {constraint} to schema {schema_type}')
    for annotation in other_metadata:
        if isinstance(annotation, at.Gt):
            return cs.no_info_after_validator_function(partial(_validators.greater_than_validator, gt=annotation.gt), schema)
        elif isinstance(annotation, at.Ge):
            return cs.no_info_after_validator_function(partial(_validators.greater_than_or_equal_validator, ge=annotation.ge), schema)
        elif isinstance(annotation, at.Lt):
            return cs.no_info_after_validator_function(partial(_validators.less_than_validator, lt=annotation.lt), schema)
        elif isinstance(annotation, at.Le):
            return cs.no_info_after_validator_function(partial(_validators.less_than_or_equal_validator, le=annotation.le), schema)
        elif isinstance(annotation, at.MultipleOf):
            return cs.no_info_after_validator_function(partial(_validators.multiple_of_validator, multiple_of=annotation.multiple_of), schema)
        elif isinstance(annotation, at.MinLen):
            return cs.no_info_after_validator_function(partial(_validators.min_length_validator, min_length=annotation.min_length), schema)
        elif isinstance(annotation, at.MaxLen):
            return cs.no_info_after_validator_function(partial(_validators.max_length_validator, max_length=annotation.max_length), schema)
        elif isinstance(annotation, at.Predicate):
            predicate_name = f'{annotation.func.__qualname__} ' if hasattr(annotation.func, '__qualname__') else ''

            def val_func(v: Any) -> Any:
                if not annotation.func(v):
                    raise PydanticCustomError('predicate_failed', f'Predicate {predicate_name}failed')
                return v
            return cs.no_info_after_validator_function(val_func, schema)
        return None
    return schema