from collections import Counter
from textwrap import dedent
from kombu.utils.encoding import bytes_to_str, safe_str
def if_not_seen(fun, obj):
    if draw.label(obj) not in seen:
        P(fun(obj))
        seen.add(draw.label(obj))