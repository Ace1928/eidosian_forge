from contextlib import contextmanager
from typing import Dict, List
@classmethod
def register_rule(cls, *, value=None, values=(), type=None, types=()):
    """
        Use it as a class decorator::

            normalizer = Normalizer('grammar', 'config')
            @normalizer.register_rule(value='foo')
            class MyRule(Rule):
                error_code = 42
        """
    values = list(values)
    types = list(types)
    if value is not None:
        values.append(value)
    if type is not None:
        types.append(type)
    if not values and (not types):
        raise ValueError('You must register at least something.')

    def decorator(rule_cls):
        for v in values:
            cls.rule_value_classes.setdefault(v, []).append(rule_cls)
        for t in types:
            cls.rule_type_classes.setdefault(t, []).append(rule_cls)
        return rule_cls
    return decorator