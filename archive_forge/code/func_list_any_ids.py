import os
from . import BioSeq
from . import Loader
from . import DBUtils
def list_any_ids(self, sql, args):
    """Return ids given a SQL statement to select for them.

        This assumes that the given SQL does a SELECT statement that
        returns a list of items. This parses them out of the 2D list
        they come as and just returns them in a list.
        """
    return self.execute_and_fetch_col0(sql, args)