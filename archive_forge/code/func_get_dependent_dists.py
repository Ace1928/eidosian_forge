from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def get_dependent_dists(dists, dist):
    """Recursively generate a list of distributions from *dists* that are
    dependent on *dist*.

    :param dists: a list of distributions
    :param dist: a distribution, member of *dists* for which we are interested
    """
    if dist not in dists:
        raise DistlibException('given distribution %r is not a member of the list' % dist.name)
    graph = make_graph(dists)
    dep = [dist]
    todo = graph.reverse_list[dist]
    while todo:
        d = todo.pop()
        dep.append(d)
        for succ in graph.reverse_list[d]:
            if succ not in dep:
                todo.append(succ)
    dep.pop(0)
    return dep