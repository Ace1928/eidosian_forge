import json
import textwrap
@classmethod
def declare_numeric_bounds(cls, schema, bounds, inclusive_bounds):
    """Given an applicable numeric schema, augment with bounds information"""
    if bounds is not None:
        low, high = bounds
        if low is not None:
            key = 'minimum' if inclusive_bounds[0] else 'exclusiveMinimum'
            schema[key] = low
        if high is not None:
            key = 'maximum' if inclusive_bounds[1] else 'exclusiveMaximum'
            schema[key] = high
    return schema