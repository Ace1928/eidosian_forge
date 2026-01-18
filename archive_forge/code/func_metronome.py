from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
@metronome.setter
def metronome(self, metronome):
    """
        Set metronome of the time signature.

        Parameters
        ----------
        metronome : int
            Metronome of the time signature.

        """
    self.data[2] = metronome