from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
def get_first_leaf(self):
    return self.children[0].get_first_leaf()