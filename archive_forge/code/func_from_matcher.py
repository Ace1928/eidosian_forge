import functools
import re
import warnings
@classmethod
def from_matcher(cls, matcher):
    if matcher == Always():
        return cls('*', _warn=False)
    elif matcher == Never():
        return cls('<0.0.0-', _warn=False)
    elif isinstance(matcher, Range):
        return cls('%s%s' % (matcher.operator, matcher.target), _warn=False)