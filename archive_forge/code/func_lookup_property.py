import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def lookup_property(property, value, positive, source=None, posix=False):
    """Looks up a property."""
    property = standardise_name(property) if property else None
    value = standardise_name(value)
    if (property, value) == ('GENERALCATEGORY', 'ASSIGNED'):
        property, value, positive = ('GENERALCATEGORY', 'UNASSIGNED', not positive)
    if posix and (not property) and (value.upper() in _POSIX_CLASSES):
        value = 'POSIX' + value
    if property:
        prop = PROPERTIES.get(property)
        if not prop:
            if not source:
                raise error('unknown property')
            raise error('unknown property', source.string, source.pos)
        prop_id, value_dict = prop
        val_id = value_dict.get(value)
        if val_id is None:
            if not source:
                raise error('unknown property value')
            raise error('unknown property value', source.string, source.pos)
        return Property(prop_id << 16 | val_id, positive)
    for property in ('GC', 'SCRIPT', 'BLOCK'):
        prop_id, value_dict = PROPERTIES.get(property)
        val_id = value_dict.get(value)
        if val_id is not None:
            return Property(prop_id << 16 | val_id, positive)
    prop = PROPERTIES.get(value)
    if prop:
        prop_id, value_dict = prop
        if set(value_dict) == _BINARY_VALUES:
            return Property(prop_id << 16 | 1, positive)
        return Property(prop_id << 16, not positive)
    if value.startswith('IS'):
        prop = PROPERTIES.get(value[2:])
        if prop:
            prop_id, value_dict = prop
            if 'YES' in value_dict:
                return Property(prop_id << 16 | 1, positive)
    for prefix, property in (('IS', 'SCRIPT'), ('IN', 'BLOCK')):
        if value.startswith(prefix):
            prop_id, value_dict = PROPERTIES.get(property)
            val_id = value_dict.get(value[2:])
            if val_id is not None:
                return Property(prop_id << 16 | val_id, positive)
    if not source:
        raise error('unknown property')
    raise error('unknown property', source.string, source.pos)