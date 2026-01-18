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
def add_imports_for_symbol(module_code_builder, symbol, source_module_name, source_name, api_name, api_version, output_module_prefix=''):
    """Add imports for the given symbol to `module_code_builder`.

  Args:
    module_code_builder: `_ModuleInitCodeBuilder` instance.
    symbol: A symbol.
    source_module_name: Module that we can import the symbol from.
    source_name: Name we can import the symbol with.
    api_name: API name. Currently, must be either `tensorflow` or `estimator`.
    api_version: API version.
    output_module_prefix: Prefix to prepend to destination module.
  """
    if api_version == 1:
        names_attr = API_ATTRS_V1[api_name].names
        constants_attr = API_ATTRS_V1[api_name].constants
    else:
        names_attr = API_ATTRS[api_name].names
        constants_attr = API_ATTRS[api_name].constants
    if source_name == constants_attr:
        for exports, name in symbol:
            for export in exports:
                dest_module, dest_name = _get_name_and_module(export)
                dest_module = _join_modules(output_module_prefix, dest_module)
                module_code_builder.add_import(None, source_module_name, name, dest_module, dest_name)
    if hasattr(symbol, '__dict__') and names_attr in symbol.__dict__:
        for export in getattr(symbol, names_attr):
            dest_module, dest_name = _get_name_and_module(export)
            dest_module = _join_modules(output_module_prefix, dest_module)
            module_code_builder.add_import(symbol, source_module_name, source_name, dest_module, dest_name)