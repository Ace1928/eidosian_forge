from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import six
from pasta.augment import import_utils
from pasta.base import ast_utils
from pasta.base import scope
def rename_external(t, old_name, new_name):
    """Rename an imported name in a module.

  This will rewrite all import statements in `tree` that reference the old
  module as well as any names in `tree` which reference the imported name. This
  may introduce new import statements, but only if necessary.

  For example, to move and rename the module `foo.bar.utils` to `foo.bar_utils`:
  > rename_external(tree, 'foo.bar.utils', 'foo.bar_utils')

  - import foo.bar.utils
  + import foo.bar_utils

  - from foo.bar import utils
  + from foo import bar_utils

  - from foo.bar import logic, utils
  + from foo.bar import logic
  + from foo import bar_utils

  Arguments:
    t: (ast.Module) Module syntax tree to perform the rename in. This will be
      updated as a result of this function call with all affected nodes changed
      and potentially new Import/ImportFrom nodes added.
    old_name: (string) Fully-qualified path of the name to replace.
    new_name: (string) Fully-qualified path of the name to update to.

  Returns:
    True if any changes were made, False otherwise.
  """
    sc = scope.analyze(t)
    if old_name not in sc.external_references:
        return False
    has_changed = False
    renames = {}
    already_changed = []
    for ref in sc.external_references[old_name]:
        if isinstance(ref.node, ast.alias):
            parent = sc.parent(ref.node)
            if isinstance(parent, ast.ImportFrom) and parent not in already_changed:
                assert _rename_name_in_importfrom(sc, parent, old_name, new_name)
                renames[old_name.rsplit('.', 1)[-1]] = new_name.rsplit('.', 1)[-1]
                already_changed.append(parent)
            else:
                ref.node.name = new_name + ref.node.name[len(old_name):]
                if not ref.node.asname:
                    renames[old_name] = new_name
            has_changed = True
        elif isinstance(ref.node, ast.ImportFrom):
            if ref.node not in already_changed:
                assert _rename_name_in_importfrom(sc, ref.node, old_name, new_name)
                renames[old_name.rsplit('.', 1)[-1]] = new_name.rsplit('.', 1)[-1]
                already_changed.append(ref.node)
                has_changed = True
    for rename_old, rename_new in six.iteritems(renames):
        _rename_reads(sc, t, rename_old, rename_new)
    return has_changed