import os
import json
from typing import Optional
import numpy as np
from ase.io.bundletrajectory import UlmBundleBackend
def read_bundle_info(name):
    """Read global info about a bundle.

    Returns (metadata, nframes)
    """
    if not os.path.isdir(name):
        raise IOError("No directory (bundle) named '%s' found." % (name,))
    metaname = os.path.join(name, 'metadata.json')
    if not os.path.isfile(metaname):
        if os.path.isfile(os.path.join(name, 'metadata')):
            raise IOError('Found obsolete metadata in unsecure Pickle format.  Refusing to load.')
        else:
            raise IOError("'{}' does not appear to be a BundleTrajectory (no {})".format(name, metaname))
    with open(metaname) as fd:
        mdata = json.load(fd)
    if 'format' not in mdata or mdata['format'] != 'BundleTrajectory':
        raise IOError("'%s' does not appear to be a BundleTrajectory" % (name,))
    if mdata['version'] != 1:
        raise IOError('Cannot manipulate BundleTrajectories with version number %s' % (mdata['version'],))
    with open(os.path.join(name, 'frames')) as fd:
        nframes = int(fd.read())
    if nframes == 0:
        raise IOError("'%s' is an empty BundleTrajectory" % (name,))
    return (mdata, nframes)