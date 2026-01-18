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
def get_subgraph_list(self):
    """Get the list of Subgraph instances.

        This method returns the list of Subgraph instances
        in the graph.
        """
    sgraph_objs = list()
    for sgraph in self.obj_dict['subgraphs']:
        obj_dict_list = self.obj_dict['subgraphs'][sgraph]
        sgraph_objs.extend([Subgraph(obj_dict=obj_d) for obj_d in obj_dict_list])
    return sgraph_objs