from fractions import Fraction
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import av
import av.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from ..core import Request
from ..core.request import URI_BYTES, InitializationError, IOMode
from ..core.v3_plugin_api import ImageProperties, PluginV3
def set_video_filter(self, filter_sequence: List[Tuple[str, Union[str, dict]]]=None, filter_graph: Tuple[dict, List]=None) -> None:
    """Set the filter(s) to use.

        This function creates a new FFMPEG filter graph to use when reading or
        writing video. In the case of reading, frames are passed through the
        filter graph before begin returned and, in case of writing, frames are
        passed through the filter before being written to the video.

        Parameters
        ----------
        filter_sequence : List[str, str, dict]
            If not None, apply the given sequence of FFmpeg filters to each
            ndimage. Check the (module-level) plugin docs for details and
            examples.
        filter_graph : (dict, List)
            If not None, apply the given graph of FFmpeg filters to each
            ndimage. The graph is given as a tuple of two dicts. The first dict
            contains a (named) set of nodes, and the second dict contains a set
            of edges between nodes of the previous dict. Check the
            (module-level) plugin docs for details and examples.

        Notes
        -----
        Changing a filter graph with lag during reading or writing will
        currently cause frames in the filter queue to be lost.

        """
    if filter_sequence is None and filter_graph is None:
        self._video_filter = None
        return
    if filter_sequence is None:
        filter_sequence = list()
    node_descriptors: Dict[str, Tuple[str, Union[str, Dict]]]
    edges: List[Tuple[str, str, int, int]]
    if filter_graph is None:
        node_descriptors, edges = (dict(), [('video_in', 'video_out', 0, 0)])
    else:
        node_descriptors, edges = filter_graph
    graph = av.filter.Graph()
    previous_node = graph.add_buffer(template=self._video_stream)
    for filter_name, argument in filter_sequence:
        if isinstance(argument, str):
            current_node = graph.add(filter_name, argument)
        else:
            current_node = graph.add(filter_name, **argument)
        previous_node.link_to(current_node)
        previous_node = current_node
    nodes = dict()
    nodes['video_in'] = previous_node
    nodes['video_out'] = graph.add('buffersink')
    for name, (filter_name, arguments) in node_descriptors.items():
        if isinstance(arguments, str):
            nodes[name] = graph.add(filter_name, arguments)
        else:
            nodes[name] = graph.add(filter_name, **arguments)
    for from_note, to_node, out_idx, in_idx in edges:
        nodes[from_note].link_to(nodes[to_node], out_idx, in_idx)
    graph.configure()

    def video_filter():
        frame = (yield None)
        while frame is not None:
            graph.push(frame)
            try:
                frame = (yield graph.pull())
            except av.error.BlockingIOError:
                frame = (yield None)
            except av.error.EOFError:
                break
        try:
            graph.push(None)
        except ValueError:
            pass
        while True:
            try:
                yield graph.pull()
            except av.error.EOFError:
                break
            except av.error.BlockingIOError:
                break
    self._video_filter = video_filter()
    self._video_filter.send(None)