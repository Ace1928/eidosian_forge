import re
from . import schema
from .search import Search
from .resources import CObject, Project, Projects, Experiment, Experiments
from .uriutil import inv_translate_uri
from .errors import ProgrammingError
def is_expand_level(element):
    return element.startswith('//') and element.strip('/') in schema.resources_types