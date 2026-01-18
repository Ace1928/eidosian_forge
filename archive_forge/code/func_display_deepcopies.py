import copy
import json
import pickle
import traceback
import parlai.utils.logging as logging
from typing import List
def display_deepcopies(self):
    """
        Display all deepcopies.
        """
    if len(self.deepcopies) == 0:
        return 'No deepcopies performed on this opt.'
    return '\n'.join((f'{i}. {loc}' for i, loc in enumerate(self.deepcopies, 1)))