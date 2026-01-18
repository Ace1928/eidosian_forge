import pickle
import re
from debian.deprecation import function_deprecated_by
def filter_packages_tags(self, package_tag_filter):
    """
        Return a collection with only those packages that match a
        filter, sharing tagsets with this one.  The filter will match
        on (package, tags).
        """
    res = DB()
    db = {}
    for pkg, _ in filter(package_tag_filter, self.db.items()):
        db[pkg] = self.db[pkg]
    res.db = db
    res.rdb = reverse(db)
    return res