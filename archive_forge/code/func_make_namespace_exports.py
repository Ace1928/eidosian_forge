import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def make_namespace_exports(components, prefix):
    export_string = ''
    for component in components:
        if not component.endswith('-*') and str(component) not in r_keywords and (str(component) not in ['setProps', 'children']):
            export_string += 'export({}{})\n'.format(prefix, component)
    rfilelist = []
    omitlist = ['utils.R', 'internal.R'] + ['{}{}.R'.format(prefix, component) for component in components]
    fnlist = []
    for script in os.listdir('R'):
        if script.endswith('.R') and script not in omitlist:
            rfilelist += [os.path.join('R', script)]
    for rfile in rfilelist:
        with open(rfile, 'r', encoding='utf-8') as script:
            s = script.read()
            s = re.sub('#.*$', '', s, flags=re.M)
            s = s.replace('\n', ' ').replace('\r', ' ')
            s = re.sub("'([^'\\\\]|\\\\'|\\\\[^'])*'", "''", s)
            s = re.sub('"([^"\\\\]|\\\\"|\\\\[^"])*"', '""', s)
            prev_len = len(s) + 1
            while len(s) < prev_len:
                prev_len = len(s)
                s = re.sub('\\(([^()]|\\(\\))*\\)', '()', s)
                s = re.sub('\\{([^{}]|\\{\\})*\\}', '{}', s)
            matches = re.findall('([^A-Za-z0-9._]|^)([A-Za-z0-9._]+)\\s*(=|<-)\\s*function', s)
            for match in matches:
                fn = match[1]
                if fn[0] != '.' and fn not in fnlist:
                    fnlist.append(fn)
    export_string += '\n'.join(('export({})'.format(function) for function in fnlist))
    return export_string