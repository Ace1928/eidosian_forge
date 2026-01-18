import re
from . import schema
from .search import Search
from .resources import CObject, Project, Projects, Experiment, Experiments
from .uriutil import inv_translate_uri
from .errors import ProgrammingError
def mtransform(paths):
    tpaths = []
    for path in paths:
        els = re.findall('/{1,2}.*?(?=/{1,2}|$)', path)
        tels = []
        ignore_path = False
        for i, curr_el in enumerate(els):
            if i + 1 < len(els):
                next_el = els[i + 1]
            else:
                next_el = None
            if is_type_level(curr_el):
                if not is_id_level(next_el):
                    if not is_singular_type_level(curr_el):
                        tels.append(curr_el)
                        tels.append('/*')
                    else:
                        tels.append(curr_el + 's')
                        tels.append('/*')
                elif not is_singular_type_level(curr_el):
                    if not is_wildid_level(next_el):
                        tels.append(curr_el.rstrip('s'))
                    else:
                        tels.append(curr_el)
                elif not is_wildid_level(next_el):
                    tels.append(curr_el)
                else:
                    tels.append(curr_el + 's')
            elif is_expand_level(curr_el):
                exp_paths = [''.join(els[:i] + [rel_path] + els[i + 1:]) for rel_path in expand_level(curr_el, path)]
                tpaths.extend(mtransform(exp_paths))
                ignore_path = True
                break
            elif is_id_level(curr_el):
                tels.append(curr_el)
            else:
                raise ProgrammingError('in %s' % path)
        if not ignore_path:
            tpaths.append(''.join(tels))
    return tpaths