from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import encoding
def decorate_v1_generator_with_v2_api_info(v1_generator, v2_generator):
    """Decorate gen1 functions in v1 API format with additional info from its v2 API format.

  Currently only the `environment` and `upgradeInfo` fields are copied over.

  Args:
    v1_generator: Generator, generating gen1 function retrieved from v1 API.
    v2_generator: Generator, generating gen1 function retrieved from v2 API.

  Yields:
    Gen1 function encoded as a dict with upgrade info decorated.
  """
    gen1_generator = sorted(itertools.chain(v1_generator, v2_generator), key=lambda f: f.name)
    for _, func_gen in itertools.groupby(gen1_generator, key=lambda f: f.name):
        func_list = list(func_gen)
        if len(func_list) < 2:
            yield func_list[0]
        else:
            v1_func, v2_func = func_list
            yield decorate_v1_function_with_v2_api_info(v1_func, v2_func)