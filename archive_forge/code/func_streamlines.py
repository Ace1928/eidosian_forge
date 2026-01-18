from abc import ABC, abstractmethod
from .header import Field
@property
def streamlines(self):
    return self.tractogram.streamlines