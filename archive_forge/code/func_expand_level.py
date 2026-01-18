import re
from . import schema
from .search import Search
from .resources import CObject, Project, Projects, Experiment, Experiments
from .uriutil import inv_translate_uri
from .errors import ProgrammingError
def expand_level(element, fullpath):

    def find_paths(element, path=[]):
        resources_dict = schema.resources_tree
        element = element.strip('/')
        paths = []
        if path == []:
            path = [element]
        init_path = path[:]
        for key in resources_dict.keys():
            path = init_path[:]
            if element in resources_dict[key]:
                path.append(key)
                look_again = find_paths(key, path)
                if look_again != []:
                    paths.extend(look_again)
                else:
                    path.reverse()
                    paths.append('/' + '/'.join(path))
        return paths
    absolute_paths = find_paths(element)
    els = re.findall('/{1,2}.*?(?=/{1,2}|$)', fullpath)
    index = els.index(element)
    if index == 0:
        return absolute_paths
    else:
        for i in range(1, 4):
            if is_type_level(els[index - i]) or is_expand_level(els[index - i]):
                parent_level = els[index - i]
                break
    if parent_level.strip('/') in schema.resources_singular:
        parent_level += 's'
    return [abspath.split(parent_level)[1] for abspath in absolute_paths if parent_level in abspath]