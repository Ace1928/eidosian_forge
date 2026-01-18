import pickle
import re
from debian.deprecation import function_deprecated_by
def filter_tags(self, tag_filter):
    """
        Return a collection with only those tags that match a
        filter, sharing package sets with this one.  The filter will match
        on the tag.
        """
    res = DB()
    rdb = {}
    for tag in filter(tag_filter, self.rdb.keys()):
        rdb[tag] = self.rdb[tag]
    res.rdb = rdb
    res.db = reverse(rdb)
    return res