import pickle
import re
from debian.deprecation import function_deprecated_by
def choose_packages_copy(self, package_iter):
    """
        Return a collection with only the packages in package_iter,
        with a copy of the tagsets of this one
        """
    res = DB()
    db = {}
    for pkg in package_iter:
        db[pkg] = self.db[pkg]
    res.db = db
    res.rdb = reverse(db)
    return res