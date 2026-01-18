from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@property
def max_wall_time(self):
    return self._max_wall_time