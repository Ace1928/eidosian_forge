import re
from . import schema
from .search import Search
from .resources import CObject, Project, Projects, Experiment, Experiments
from .uriutil import inv_translate_uri
from .errors import ProgrammingError
def is_id_level(element):
    return element is not None and element.strip('/') not in schema.resources_types