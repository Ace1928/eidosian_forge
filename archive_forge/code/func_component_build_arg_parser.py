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
def component_build_arg_parser():
    parser = argparse.ArgumentParser(prog='dash-generate-components', formatter_class=_CombinedFormatter, description='Generate dash components by extracting the metadata using react-docgen. Then map the metadata to Python classes.')
    parser.add_argument('components_source', help='React components source directory.')
    parser.add_argument('project_shortname', help='Name of the project to export the classes files.')
    parser.add_argument('-p', '--package-info-filename', default='package.json', help='The filename of the copied `package.json` to `project_shortname`')
    parser.add_argument('-i', '--ignore', default='^_', help='Files/directories matching the pattern will be ignored')
    parser.add_argument('--r-prefix', help='Specify a prefix for Dash for R component names, write components to R dir, create R package.')
    parser.add_argument('--r-depends', default='', help='Specify a comma-separated list of R packages to be inserted into the Depends field of the DESCRIPTION file.')
    parser.add_argument('--r-imports', default='', help='Specify a comma-separated list of R packages to be inserted into the Imports field of the DESCRIPTION file.')
    parser.add_argument('--r-suggests', default='', help='Specify a comma-separated list of R packages to be inserted into the Suggests field of the DESCRIPTION file.')
    parser.add_argument('--jl-prefix', help='Specify a prefix for Dash for R component names, write components to R dir, create R package.')
    parser.add_argument('-k', '--keep-prop-order', default=None, help="Specify a comma-separated list of components which will use the prop order described in the component proptypes instead of alphabetically reordered props. Pass the 'ALL' keyword to have every component retain its original prop order.")
    parser.add_argument('--max-props', type=int, default=250, help='Specify the max number of props to list in the component signature. More props will still be shown in the docstring, and will still work when provided as kwargs to the component. Python <3.7 only supports 255 args, but you may also want to reduce further for improved readability at the expense of auto-completion for the later props. Use 0 to include all props.')
    return parser