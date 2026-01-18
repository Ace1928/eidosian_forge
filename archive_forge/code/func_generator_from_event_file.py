import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
def generator_from_event_file(event_file):
    """Returns a generator that yields events from an event file."""
    return event_file_loader.LegacyEventFileLoader(event_file).Load()