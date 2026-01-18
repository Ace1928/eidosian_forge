from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
@property
def rules_alpha(self):
    return self.split_alpha_beta()[0]