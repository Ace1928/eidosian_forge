from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
@property
def run_name(self):
    return self._run_name