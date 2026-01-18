import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
def create_primary_api_files(output_files, packages, packages_to_ignore, root_init_template, output_dir, output_package, api_name, api_version, compat_api_versions, compat_init_templates, lazy_loading=_LAZY_LOADING, use_relative_imports=False):
    """Creates __init__.py files for the Python API.

  Args:
    output_files: List of __init__.py file paths to create.
    packages: Base python packages containing python with target tf_export
      decorators.
    packages_to_ignore: python packages to be ignored when checking for
      tf_export decorators.
    root_init_template: Template for top-level __init__.py file. "# API IMPORTS
      PLACEHOLDER" comment in the template file will be replaced with imports.
    output_dir: output API root directory.
    output_package: Base output package where generated API will be added.
    api_name: API you want to generate (e.g. `tensorflow` or `estimator`).
    api_version: API version to generate (`v1` or `v2`).
    compat_api_versions: Additional API versions to generate in compat/
      subdirectory.
    compat_init_templates: List of templates for top level compat init files in
      the same order as compat_api_versions.
    lazy_loading: Boolean flag. If True, a lazy loading `__init__.py` file is
      produced and if `False`, static imports are used.
    use_relative_imports: True if we should use relative imports when import
      submodules.

  Raises:
    ValueError: if output_files list is missing a required file.
  """
    module_name_to_file_path = {}
    for output_file in output_files:
        module_name = get_module(os.path.dirname(output_file), output_dir)
        module_name_to_file_path[module_name] = os.path.normpath(output_file)
    for module, file_path in module_name_to_file_path.items():
        if not os.path.isdir(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        open(file_path, 'a').close()
    module_text_map, deprecation_footer_map, root_module_footer = get_api_init_text(packages, packages_to_ignore, output_package, api_name, api_version, compat_api_versions, lazy_loading, use_relative_imports)
    missing_output_files = []
    root_module = ''
    compat_module_to_template = {_COMPAT_MODULE_TEMPLATE % v: t for v, t in zip(compat_api_versions, compat_init_templates)}
    for v in compat_api_versions:
        compat_module_to_template.update({_SUBCOMPAT_MODULE_TEMPLATE % (v, vs): t for vs, t in zip(compat_api_versions, compat_init_templates)})
    for module, text in module_text_map.items():
        if module not in module_name_to_file_path:
            module_file_path = '"%s/__init__.py"' % module.replace('.', '/')
            missing_output_files.append(module_file_path)
            continue
        contents = ''
        if module == root_module and root_init_template:
            with open(root_init_template, 'r') as root_init_template_file:
                contents = root_init_template_file.read()
                contents = contents.replace('# API IMPORTS PLACEHOLDER', text)
                contents = contents.replace('# __all__ PLACEHOLDER', root_module_footer)
        elif module in compat_module_to_template:
            with open(compat_module_to_template[module], 'r') as init_template_file:
                contents = init_template_file.read()
                contents = contents.replace('# API IMPORTS PLACEHOLDER', text)
        else:
            contents = _GENERATED_FILE_HEADER % get_module_docstring(module, packages[0], api_name) + text + _GENERATED_FILE_FOOTER
        if module in deprecation_footer_map:
            if '# WRAPPER_PLACEHOLDER' in contents:
                contents = contents.replace('# WRAPPER_PLACEHOLDER', deprecation_footer_map[module])
            else:
                contents += deprecation_footer_map[module]
        with open(module_name_to_file_path[module], 'w') as fp:
            fp.write(contents)
    if missing_output_files:
        missing_files = ',\n'.join(sorted(missing_output_files))
        raise ValueError(f'Missing outputs for genrule:\n{missing_files}. Be sure to add these targets to tensorflow/python/tools/api/generator/api_init_files_v1.bzl and tensorflow/python/tools/api/generator/api_init_files.bzl (tensorflow repo), keras/api/api_init_files.bzl (keras repo), or tensorflow_estimator/python/estimator/api/api_gen.bzl (estimator repo)')