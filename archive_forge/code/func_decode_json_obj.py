from abc import ABCMeta, abstractmethod
from nltk import jsontags
@classmethod
def decode_json_obj(cls, obj):
    return cls(obj['templateid'], obj['original'], obj['replacement'], tuple((tuple(feat) for feat in obj['conditions'])))