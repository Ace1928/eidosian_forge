import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def del_edge(self, src_or_list, dst=None, index=None):
    """Delete an edge from the graph.

        Given an edge's (source, destination) node names all
        matching edges(s) will be deleted if 'index' is not
        specified or set to None.
        If there are several matching edges and 'index' is
        given, only the edge in that position will be deleted.

        'index' should be an integer specifying the position
        of the edge to delete. If index is larger than the
        number of matching edges, no action is taken.

        If edges are deleted it returns True. If no action
        is taken it returns False.
        """
    if isinstance(src_or_list, (list, tuple)):
        if dst is not None and isinstance(dst, int):
            index = dst
        src, dst = src_or_list
    else:
        src, dst = (src_or_list, dst)
    if isinstance(src, Node):
        src = src.get_name()
    if isinstance(dst, Node):
        dst = dst.get_name()
    if (src, dst) in self.obj_dict['edges']:
        if index is not None and index < len(self.obj_dict['edges'][src, dst]):
            del self.obj_dict['edges'][src, dst][index]
            return True
        else:
            del self.obj_dict['edges'][src, dst]
            return True
    return False