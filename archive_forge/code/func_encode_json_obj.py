from abc import ABCMeta, abstractmethod
from nltk import jsontags
def encode_json_obj(self):
    return {'templateid': self.templateid, 'original': self.original_tag, 'replacement': self.replacement_tag, 'conditions': self._conditions}