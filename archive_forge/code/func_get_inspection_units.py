import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
def get_inspection_units(logdir='', event_file='', tag=''):
    """Returns a list of InspectionUnit objects given either logdir or
    event_file.

    If logdir is given, the number of InspectionUnits should equal the
    number of directories or subdirectories that contain event files.

    If event_file is given, the number of InspectionUnits should be 1.

    Args:
      logdir: A log directory that contains event files.
      event_file: Or, a particular event file path.
      tag: An optional tag name to query for.

    Returns:
      A list of InspectionUnit objects.
    """
    if logdir:
        subdirs = io_wrapper.GetLogdirSubdirectories(logdir)
        inspection_units = []
        for subdir in subdirs:
            generator = itertools.chain(*[generator_from_event_file(os.path.join(subdir, f)) for f in tf.io.gfile.listdir(subdir) if io_wrapper.IsTensorFlowEventsFile(os.path.join(subdir, f))])
            inspection_units.append(InspectionUnit(name=subdir, generator=generator, field_to_obs=get_field_to_observations_map(generator, tag)))
        if inspection_units:
            print('Found event files in:\n{}\n'.format('\n'.join([u.name for u in inspection_units])))
        elif io_wrapper.IsTensorFlowEventsFile(logdir):
            print('It seems that {} may be an event file instead of a logdir. If this is the case, use --event_file instead of --logdir to pass it in.'.format(logdir))
        else:
            print('No event files found within logdir {}'.format(logdir))
        return inspection_units
    elif event_file:
        generator = generator_from_event_file(event_file)
        return [InspectionUnit(name=event_file, generator=generator, field_to_obs=get_field_to_observations_map(generator, tag))]
    return []