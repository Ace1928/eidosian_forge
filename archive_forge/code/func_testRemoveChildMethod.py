from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pasta
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import test_utils
def testRemoveChildMethod(self):
    src = 'class C():\n  def f(x):\n    return x + 2\n  def g(x):\n    return x + 3\n'
    tree = pasta.parse(src)
    class_node = tree.body[0]
    meth1_node = class_node.body[0]
    ast_utils.remove_child(class_node, meth1_node)
    result = pasta.dump(tree)
    expected = 'class C():\n  def g(x):\n    return x + 3\n'
    self.assertEqual(result, expected)