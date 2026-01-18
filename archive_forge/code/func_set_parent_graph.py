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
def set_parent_graph(self, parent_graph):
    self.obj_dict['parent_graph'] = parent_graph
    for k in self.obj_dict['nodes']:
        obj_list = self.obj_dict['nodes'][k]
        for obj in obj_list:
            obj['parent_graph'] = parent_graph
    for k in self.obj_dict['edges']:
        obj_list = self.obj_dict['edges'][k]
        for obj in obj_list:
            obj['parent_graph'] = parent_graph
    for k in self.obj_dict['subgraphs']:
        obj_list = self.obj_dict['subgraphs'][k]
        for obj in obj_list:
            Graph(obj_dict=obj).set_parent_graph(parent_graph)