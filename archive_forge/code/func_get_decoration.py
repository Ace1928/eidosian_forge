import sys
import os
import re
import warnings
import types
import unicodedata
def get_decoration(self):
    if not self.decoration:
        self.decoration = decoration()
        index = self.first_child_not_matching_class(Titular)
        if index is None:
            self.append(self.decoration)
        else:
            self.insert(index, self.decoration)
    return self.decoration