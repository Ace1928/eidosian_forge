import pathlib
import pickle
import unittest
from traits.testing.optional_dependencies import (
def find_pickles():
    """
    Iterate over the pickle files in the test_data directory.

    Skip files that correspond to a protocol not supported with
    the current version of Python.

    Yields paths to pickle files.
    """
    pickle_directory = pathlib.Path(pkg_resources.resource_filename('traits.tests', 'test-data/historical-pickles'))
    for pickle_path in pickle_directory.glob('*.pkl'):
        header, _, protocol, _ = pickle_path.name.split('-', maxsplit=3)
        if header != 'hipt':
            continue
        if not protocol.startswith('p'):
            raise RuntimeError("Can't interpret protocol: {}".format(protocol))
        protocol = int(protocol[1:])
        if protocol > pickle.HIGHEST_PROTOCOL:
            continue
        yield pickle_path