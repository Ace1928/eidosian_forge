import abc
from taskflow import exceptions as exc
from taskflow.persistence import base
from taskflow.persistence import models
@property
def atom_path(self):
    return self._atom_path