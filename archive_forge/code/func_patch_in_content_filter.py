import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def patch_in_content_filter(self):

    def new_stack(tree, path=None, file_id=None):
        if path.endswith('.txt'):
            return [ContentFilter(_swapcase, _swapcase)]
        else:
            return []
    self.overrideAttr(WorkingTree, '_content_filter_stack', new_stack)