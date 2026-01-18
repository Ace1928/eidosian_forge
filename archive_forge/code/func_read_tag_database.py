import pickle
import re
from debian.deprecation import function_deprecated_by
def read_tag_database(input_data):
    """Read the tag database, returning a pkg->tags dictionary"""
    db = {}
    for pkgs, tags in parse_tags(input_data):
        for p in pkgs:
            db[p] = tags.copy()
    return db