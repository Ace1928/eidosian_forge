import pickle
import re
from debian.deprecation import function_deprecated_by
def qwrite(self, file):
    """Quickly write the data to a pickled file"""
    pickle.dump(self.db, file)
    pickle.dump(self.rdb, file)