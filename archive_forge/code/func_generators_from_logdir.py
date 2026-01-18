import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
def generators_from_logdir(logdir):
    """Returns a list of event generators for subdirectories with event files.

    The number of generators returned should equal the number of directories
    within logdir that contain event files. If only logdir contains event files,
    returns a list of length one.

    Args:
      logdir: A log directory that contains event files.

    Returns:
      List of event generators for each subdirectory with event files.
    """
    subdirs = io_wrapper.GetLogdirSubdirectories(logdir)
    generators = [itertools.chain(*[generator_from_event_file(os.path.join(subdir, f)) for f in tf.io.gfile.listdir(subdir) if io_wrapper.IsTensorFlowEventsFile(os.path.join(subdir, f))]) for subdir in subdirs]
    return generators