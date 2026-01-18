from parlai.core.dict import DictionaryAgent
from abc import ABC, abstractmethod
def is_prebuilt(self):
    """
        Indicates whether the dictionary is fixed, and does not require building.
        """
    return True