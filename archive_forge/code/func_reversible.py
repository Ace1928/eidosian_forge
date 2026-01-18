from django.db import router
from .base import Operation
@property
def reversible(self):
    return self.reverse_code is not None