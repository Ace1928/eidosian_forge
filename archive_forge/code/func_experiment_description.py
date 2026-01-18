from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@property
def experiment_description(self):
    return self._experiment_description