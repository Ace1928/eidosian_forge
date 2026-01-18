import pickle
import re
from debian.deprecation import function_deprecated_by
def filter_packages(self, package_filter):
    """
        Return a collection with only those packages that match a
        filter, sharing tagsets with this one.  The filter will match
        on the package.
        """
    res = DB()
    db = {}
    for pkg in filter(package_filter, self.db.keys()):
        db[pkg] = self.db[pkg]
    res.db = db
    res.rdb = reverse(db)
    return res