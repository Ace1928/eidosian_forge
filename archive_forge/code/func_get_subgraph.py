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
def get_subgraph(self, name):
    """Retrieved a subgraph from the graph.

        Given a subgraph's name the corresponding
        Subgraph instance will be returned.

        If one or more subgraphs exist with the same name, a list of
        Subgraph instances is returned.
        An empty list is returned otherwise.
        """
    match = list()
    if name in self.obj_dict['subgraphs']:
        sgraphs_obj_dict = self.obj_dict['subgraphs'].get(name)
        for obj_dict_list in sgraphs_obj_dict:
            match.append(Subgraph(obj_dict=obj_dict_list))
    return match