from collections import OrderedDict
import json
import sys
import subprocess
import shlex
import os
import argparse
import shutil
import functools
import pkg_resources
import yaml
from ._r_components_generation import write_class_file
from ._r_components_generation import generate_exports
from ._py_components_generation import generate_class_file
from ._py_components_generation import generate_imports
from ._py_components_generation import generate_classes_files
from ._jl_components_generation import generate_struct_file
from ._jl_components_generation import generate_module
def generate_components(components_source, project_shortname, package_info_filename='package.json', ignore='^_', rprefix=None, rdepends='', rimports='', rsuggests='', jlprefix=None, metadata=None, keep_prop_order=None, max_props=None):
    project_shortname = project_shortname.replace('-', '_').rstrip('/\\')
    is_windows = sys.platform == 'win32'
    extract_path = pkg_resources.resource_filename('dash', 'extract-meta.js')
    reserved_patterns = '|'.join((f'^{p}$' for p in reserved_words))
    os.environ['NODE_PATH'] = 'node_modules'
    shutil.copyfile('package.json', os.path.join(project_shortname, package_info_filename))
    if not metadata:
        env = os.environ.copy()
        env['MODULES_PATH'] = os.path.abspath('./node_modules')
        cmd = shlex.split(f'node {extract_path} "{ignore}" "{reserved_patterns}" {components_source}', posix=not is_windows)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=is_windows, env=env)
        out, err = proc.communicate()
        status = proc.poll()
        if err:
            print(err.decode(), file=sys.stderr)
        if not out:
            print(f'Error generating metadata in {project_shortname} (status={status})', file=sys.stderr)
            sys.exit(1)
        metadata = safe_json_loads(out.decode('utf-8'))
    py_generator_kwargs = {}
    if keep_prop_order is not None:
        keep_prop_order = [component.strip(' ') for component in keep_prop_order.split(',')]
        py_generator_kwargs['prop_reorder_exceptions'] = keep_prop_order
    if max_props:
        py_generator_kwargs['max_props'] = max_props
    generator_methods = [functools.partial(generate_class_file, **py_generator_kwargs)]
    if rprefix is not None or jlprefix is not None:
        with open('package.json', 'r', encoding='utf-8') as f:
            pkg_data = safe_json_loads(f.read())
    if rprefix is not None:
        if not os.path.exists('man'):
            os.makedirs('man')
        if not os.path.exists('R'):
            os.makedirs('R')
        if os.path.isfile('dash-info.yaml'):
            with open('dash-info.yaml', encoding='utf-8') as yamldata:
                rpkg_data = yaml.safe_load(yamldata)
        else:
            rpkg_data = None
        generator_methods.append(functools.partial(write_class_file, prefix=rprefix, rpkg_data=rpkg_data))
    if jlprefix is not None:
        generator_methods.append(functools.partial(generate_struct_file, prefix=jlprefix))
    components = generate_classes_files(project_shortname, metadata, *generator_methods)
    with open(os.path.join(project_shortname, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    generate_imports(project_shortname, components)
    if rprefix is not None:
        generate_exports(project_shortname, components, metadata, pkg_data, rpkg_data, rprefix, rdepends, rimports, rsuggests)
    if jlprefix is not None:
        generate_module(project_shortname, components, metadata, pkg_data, jlprefix)