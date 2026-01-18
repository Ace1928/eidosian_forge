from breezy import rules
from breezy.tests.per_tree import TestCaseWithTree
def make_tree_with_rules(self, text):
    tree = self.make_branch_and_tree('.')
    if text is not None:
        self.fail('No method for in-tree rules agreed on yet.')
        text_utf8 = text.encode('utf-8')
        self.build_tree_contents([(rules.RULES_TREE_FILENAME, text_utf8)])
        tree.add(rules.RULES_TREE_FILENAME)
        tree.commit('add rules file')
    result = self._convert_tree(tree)
    result.lock_read()
    self.addCleanup(result.unlock)
    return result