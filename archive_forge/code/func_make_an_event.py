import abc
import collections
import threading
from automaton import exceptions as machine_excp
from automaton import machines
import fasteners
import futurist
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.action_engine import executor
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.types import failure as ft
from taskflow.utils import schema_utils as su
def make_an_event(new_state):
    """Turns a new/target state into an event name."""
    return ('on_%s' % new_state).lower()