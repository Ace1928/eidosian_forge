from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
@classmethod
def is_transition_valid(cls, source, target):
    if target == cls.READY:
        return source == cls.UNINITIALIZED or source == cls.COMMITTING_TRANSACTION or source == cls.ABORTING_TRANSACTION
    elif target == cls.IN_TRANSACTION:
        return source == cls.READY
    elif target == cls.COMMITTING_TRANSACTION:
        return source == cls.IN_TRANSACTION
    elif target == cls.ABORTING_TRANSACTION:
        return source == cls.IN_TRANSACTION or source == cls.ABORTABLE_ERROR
    elif target == cls.ABORTABLE_ERROR or target == cls.FATAL_ERROR:
        return True