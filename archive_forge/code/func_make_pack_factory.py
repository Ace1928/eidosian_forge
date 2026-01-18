import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def make_pack_factory(graph, delta, keylength):
    """Create a factory for creating a pack based VersionedFiles.

    This is only functional enough to run interface tests, it doesn't try to
    provide a full pack environment.

    :param graph: Store a graph.
    :param delta: Delta compress contents.
    :param keylength: How long should keys be.
    """

    def factory(transport):
        parents = graph or delta
        ref_length = 0
        if graph:
            ref_length += 1
        if delta:
            ref_length += 1
            max_delta_chain = 200
        else:
            max_delta_chain = 0
        graph_index = _mod_index.InMemoryGraphIndex(reference_lists=ref_length, key_elements=keylength)
        stream = transport.open_write_stream('newpack')
        writer = pack.ContainerWriter(stream.write)
        writer.begin()
        index = _KnitGraphIndex(graph_index, lambda: True, parents=parents, deltas=delta, add_callback=graph_index.add_nodes)
        access = pack_repo._DirectPackAccess({})
        access.set_writer(writer, graph_index, (transport, 'newpack'))
        result = KnitVersionedFiles(index, access, max_delta_chain=max_delta_chain)
        result.stream = stream
        result.writer = writer
        return result
    return factory