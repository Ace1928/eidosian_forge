import os
from traitlets import Unicode
from ipywidgets import DOMWidget
from ._version import semver
def register_component_from_file(name, file_name, relative_to_file=None):
    if name is None:
        name = file_name
        file_name = relative_to_file
        relative_to_file = None
    if relative_to_file:
        file_name = os.path.join(os.path.dirname(relative_to_file), file_name)
    with open(file_name) as f:
        vue_component_files[os.path.abspath(file_name)] = name
        register_component_from_string(name, f.read())