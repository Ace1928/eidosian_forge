import warnings
import re
from bs4.builder import (
from bs4.element import (
import html5lib
from html5lib.constants import (
from bs4.element import (
def create_treebuilder(self, namespaceHTMLElements):
    self.underlying_builder = TreeBuilderForHtml5lib(namespaceHTMLElements, self.soup, store_line_numbers=self.store_line_numbers)
    return self.underlying_builder