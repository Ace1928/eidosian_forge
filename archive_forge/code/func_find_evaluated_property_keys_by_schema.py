from collections.abc import Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit
import itertools
import re
def find_evaluated_property_keys_by_schema(validator, instance, schema):
    """
    Get all keys of items that get evaluated under the current schema.

    Covers all keywords related to unevaluatedProperties: properties,
    additionalProperties, unevaluatedProperties, patternProperties,
    dependentSchemas, allOf, oneOf, anyOf, if, then, else
    """
    if validator.is_type(schema, 'boolean'):
        return []
    evaluated_keys = []
    ref = schema.get('$ref')
    if ref is not None:
        resolved = validator._resolver.lookup(ref)
        evaluated_keys.extend(find_evaluated_property_keys_by_schema(validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents))
    dynamicRef = schema.get('$dynamicRef')
    if dynamicRef is not None:
        resolved = validator._resolver.lookup(dynamicRef)
        evaluated_keys.extend(find_evaluated_property_keys_by_schema(validator.evolve(schema=resolved.contents, _resolver=resolved.resolver), instance, resolved.contents))
    for keyword in ['properties', 'additionalProperties', 'unevaluatedProperties']:
        if keyword in schema:
            schema_value = schema[keyword]
            if validator.is_type(schema_value, 'boolean') and schema_value:
                evaluated_keys += instance.keys()
            elif validator.is_type(schema_value, 'object'):
                for property in schema_value:
                    if property in instance:
                        evaluated_keys.append(property)
    if 'patternProperties' in schema:
        for property in instance:
            for pattern in schema['patternProperties']:
                if re.search(pattern, property):
                    evaluated_keys.append(property)
    if 'dependentSchemas' in schema:
        for property, subschema in schema['dependentSchemas'].items():
            if property not in instance:
                continue
            evaluated_keys += find_evaluated_property_keys_by_schema(validator, instance, subschema)
    for keyword in ['allOf', 'oneOf', 'anyOf']:
        if keyword in schema:
            for subschema in schema[keyword]:
                errs = next(validator.descend(instance, subschema), None)
                if errs is None:
                    evaluated_keys += find_evaluated_property_keys_by_schema(validator, instance, subschema)
    if 'if' in schema:
        if validator.evolve(schema=schema['if']).is_valid(instance):
            evaluated_keys += find_evaluated_property_keys_by_schema(validator, instance, schema['if'])
            if 'then' in schema:
                evaluated_keys += find_evaluated_property_keys_by_schema(validator, instance, schema['then'])
        elif 'else' in schema:
            evaluated_keys += find_evaluated_property_keys_by_schema(validator, instance, schema['else'])
    return evaluated_keys