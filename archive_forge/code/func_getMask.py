import re
import itertools
@classmethod
def getMask(cls, maskPattern):
    return cls.maskPattern[maskPattern]