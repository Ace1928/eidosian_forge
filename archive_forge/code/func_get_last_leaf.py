from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
def get_last_leaf(self):
    return self.children[-1].get_last_leaf()