from __future__ import absolute_import
import os
from .. import Utils
def copy_inherited_directives(outer_directives, **new_directives):
    new_directives_out = dict(outer_directives)
    for name in ('test_assert_path_exists', 'test_fail_if_path_exists', 'test_assert_c_code_has', 'test_fail_if_c_code_has'):
        new_directives_out.pop(name, None)
    new_directives_out.update(new_directives)
    return new_directives_out