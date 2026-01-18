import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def limit_memory_usage(quota_or_engine, *args):
    if isinstance(quota_or_engine, int):
        quota = quota_or_engine
    else:
        quota = get_memory_quota(quota_or_engine)
    if quota <= 0:
        return
    total = 0
    for t in args:
        total += t[0] * sys.getsizeof(t[1], 0)
        if total > quota:
            raise exceptions.MemoryQuotaExceededException()