from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def add_to_preamble(self, *pa):
    self.preamble.extend(pa)