import abc
from taskflow import exceptions as exc
from taskflow.persistence import base
from taskflow.persistence import models
@property
def flow_path(self):
    return self._flow_path