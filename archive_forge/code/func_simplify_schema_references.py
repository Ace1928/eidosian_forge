from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def simplify_schema_references(schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
    definitions: dict[str, core_schema.CoreSchema] = {}
    ref_counts: dict[str, int] = defaultdict(int)
    involved_in_recursion: dict[str, bool] = {}
    current_recursion_ref_count: dict[str, int] = defaultdict(int)

    def collect_refs(s: core_schema.CoreSchema, recurse: Recurse) -> core_schema.CoreSchema:
        if s['type'] == 'definitions':
            for definition in s['definitions']:
                ref = get_ref(definition)
                assert ref is not None
                if ref not in definitions:
                    definitions[ref] = definition
                recurse(definition, collect_refs)
            return recurse(s['schema'], collect_refs)
        else:
            ref = get_ref(s)
            if ref is not None:
                new = recurse(s, collect_refs)
                new_ref = get_ref(new)
                if new_ref:
                    definitions[new_ref] = new
                return core_schema.definition_reference_schema(schema_ref=ref)
            else:
                return recurse(s, collect_refs)
    schema = walk_core_schema(schema, collect_refs)

    def count_refs(s: core_schema.CoreSchema, recurse: Recurse) -> core_schema.CoreSchema:
        if s['type'] != 'definition-ref':
            return recurse(s, count_refs)
        ref = s['schema_ref']
        ref_counts[ref] += 1
        if ref_counts[ref] >= 2:
            if current_recursion_ref_count[ref] != 0:
                involved_in_recursion[ref] = True
            return s
        current_recursion_ref_count[ref] += 1
        recurse(definitions[ref], count_refs)
        current_recursion_ref_count[ref] -= 1
        return s
    schema = walk_core_schema(schema, count_refs)
    assert all((c == 0 for c in current_recursion_ref_count.values())), 'this is a bug! please report it'

    def can_be_inlined(s: core_schema.DefinitionReferenceSchema, ref: str) -> bool:
        if ref_counts[ref] > 1:
            return False
        if involved_in_recursion.get(ref, False):
            return False
        if 'serialization' in s:
            return False
        if 'metadata' in s:
            metadata = s['metadata']
            for k in ('pydantic_js_functions', 'pydantic_js_annotation_functions', 'pydantic.internal.union_discriminator'):
                if k in metadata:
                    return False
        return True

    def inline_refs(s: core_schema.CoreSchema, recurse: Recurse) -> core_schema.CoreSchema:
        if s['type'] == 'definition-ref':
            ref = s['schema_ref']
            if can_be_inlined(s, ref):
                new = definitions.pop(ref)
                ref_counts[ref] -= 1
                if 'serialization' in s:
                    new['serialization'] = s['serialization']
                s = recurse(new, inline_refs)
                return s
            else:
                return recurse(s, inline_refs)
        else:
            return recurse(s, inline_refs)
    schema = walk_core_schema(schema, inline_refs)
    def_values = [v for v in definitions.values() if ref_counts[v['ref']] > 0]
    if def_values:
        schema = core_schema.definitions_schema(schema=schema, definitions=def_values)
    return schema