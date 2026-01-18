import re
from . import schema
from .search import Search
from .resources import CObject, Project, Projects, Experiment, Experiments
from .uriutil import inv_translate_uri
from .errors import ProgrammingError
def group_paths(paths):
    groups = {}
    for path in paths:
        resources = [el for el in re.findall('/{1,2}.*?(?=/{1,2}|$)', path) if el.strip('/') in schema.resources_types and el.strip('/') not in ['files', 'file']]
        if len(resources) == 1:
            groups.setdefault(resources[0], set()).add(path)
            continue
        for alt_path in paths:
            if alt_path.endswith(path):
                alt_rsc = [el for el in re.findall('/{1,2}.*?(?=/{1,2}|$)', alt_path) if el.strip('/') in schema.resources_types and el.strip('/') not in ['files', 'file']]
                if alt_rsc[-1].strip('/') in ['files', 'file', 'resources', 'resource'] + list(schema.rest_translation.keys()):
                    groups.setdefault(alt_rsc[-2] + alt_rsc[-1], set()).add(alt_path)
                else:
                    groups.setdefault(alt_rsc[-1], set()).add(alt_path)
    return groups