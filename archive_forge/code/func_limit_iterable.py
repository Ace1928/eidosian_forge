import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def limit_iterable(iterable, limit_or_engine):
    if isinstance(limit_or_engine, int):
        max_count = limit_or_engine
    else:
        max_count = get_max_collection_size(limit_or_engine)
    if isinstance(iterable, (SequenceType, MappingType, SetType)):
        if 0 <= max_count < len(iterable):
            raise exceptions.CollectionTooLargeException(max_count)
        return iterable

    def limiting_iterator():
        for i, t in enumerate(iterable):
            if 0 <= max_count <= i:
                raise exceptions.CollectionTooLargeException(max_count)
            yield t
    return limiting_iterator()