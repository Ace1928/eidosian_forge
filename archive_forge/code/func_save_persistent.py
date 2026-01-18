import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
def save_persistent():
    try:
        source, clone = self._atomdetail_by_name(self.injector_name, expected_type=models.TaskDetail, clone=True)
    except exceptions.NotFound:
        source = self._create_atom_detail(self.injector_name, models.TaskDetail, atom_state=None)
        fd_source, fd_clone = self._fetch_flowdetail(clone=True)
        fd_clone.add(source)
        self._with_connection(self._save_flow_detail, fd_source, fd_clone)
        self._atom_name_to_uuid[source.name] = source.uuid
        clone = source
        clone.results = dict(pairs)
        clone.state = states.SUCCESS
    else:
        clone.results.update(pairs)
    result = self._with_connection(self._save_atom_detail, source, clone)
    return (self.injector_name, result.results.keys())