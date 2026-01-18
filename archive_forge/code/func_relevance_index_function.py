import pickle
import re
from debian.deprecation import function_deprecated_by
def relevance_index_function(full, sub):
    return lambda tag: float(sub.card(tag) ** 2) / float(full.card(tag))