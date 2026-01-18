import re
import time
from typing import Optional
import IPython.display as disp
from ..trainer_callback import TrainerCallback
from ..trainer_utils import IntervalStrategy, has_length
def remove_child(self):
    """
        Closes the child progress bar.
        """
    self.child_bar = None
    self.display()